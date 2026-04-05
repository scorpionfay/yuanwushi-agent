"""
agent.py — Teacher AI agent for 元吾氏 (powered by Claude + semantic RAG).

Usage:
    python agent.py           # start chat (requires pre-built knowledge base)
    python agent.py --build   # crawl websites + extract PDFs, build knowledge base
"""

import json
import os
import re
import datetime
import sys
import anthropic
import requests
import pypdf
import numpy as np
import faiss
import jieba
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path

FILES_DB = "teacher_files.json"
EMBEDDINGS_DB = "teacher_embeddings.npy"
FEEDBACK_LOG = "feedback_log.jsonl"
PDF_DIR = Path("files_to_feed")
WEBSITE_URLS = [
    "https://awakenology.org/Chinese/",
    "https://www.awakenology.org/English/",
]
TEACHER_NAME = "元吾氏"
MODEL = "claude-haiku-4-5-20251001"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 600  # max chars per chunk
TOP_K = 8               # chunks to retrieve per query (each method)
RRF_K = 60              # Reciprocal Rank Fusion constant
MAX_HISTORY_TURNS = 10  # keep last N turns in API history
MIN_RETRIEVAL_SCORE = 0.2  # cosine similarity threshold; hybrid RRF raises effective bar


# ── Text processing ────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """Split text at paragraph/sentence boundaries to avoid mid-sentence cuts."""
    # Split into paragraphs first; fall back to sentences for dense text
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    if len(paragraphs) < 3:
        paragraphs = [s.strip() for s in re.split(r'(?<=[。！？.!?])\s+', text) if s.strip()]

    chunks = []
    buffer: list[str] = []
    buffer_len = 0

    def flush(buf: list[str]) -> None:
        joined = "\n\n".join(buf).strip()
        if len(joined) > 50:
            chunks.append({"text": joined, "source": source})

    for para in paragraphs:
        # Paragraph itself exceeds limit — split by sentences
        if len(para) > CHUNK_SIZE:
            if buffer:
                flush(buffer)
                buffer, buffer_len = [], 0
            sents = re.split(r'(?<=[。！？.!?])\s*', para)
            sub: list[str] = []
            sub_len = 0
            for sent in sents:
                if sub_len + len(sent) > CHUNK_SIZE and sub:
                    flush(sub)
                    sub, sub_len = sub[-1:], len(sub[-1])  # last sentence as overlap
                sub.append(sent)
                sub_len += len(sent)
            if sub:
                flush(sub)
            continue

        # Adding this paragraph would exceed limit — flush first
        if buffer_len + len(para) > CHUNK_SIZE and buffer:
            flush(buffer)
            buffer = buffer[-1:]          # keep last paragraph as overlap
            buffer_len = len(buffer[0]) if buffer else 0

        buffer.append(para)
        buffer_len += len(para)

    if buffer:
        flush(buffer)

    return chunks


def build_chunks(pdf_texts: list[dict], page_texts: list[dict]) -> list[dict]:
    chunks = []
    for pdf in pdf_texts:
        chunks.extend(chunk_text(pdf["text"], pdf["filename"]))
    for page in page_texts:
        chunks.extend(chunk_text(page["text"], page.get("title", page["url"])))
    return chunks


# ── Hybrid retrieval (semantic + BM25 → RRF) ──────────────────────────────────

def load_embed_model() -> SentenceTransformer:
    print(f"📡 正在加载语义检索模型（{EMBED_MODEL}）…", end="", flush=True)
    model = SentenceTransformer(EMBED_MODEL)
    print(" 完成")
    return model


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    print("   正在构建 BM25 索引…", end="", flush=True)
    corpus = [list(jieba.cut(c["text"])) for c in chunks]
    bm25 = BM25Okapi(corpus)
    print(" 完成")
    return bm25


def _rrf_combine(ranked_lists: list[list[int]]) -> list[int]:
    """Reciprocal Rank Fusion: merge multiple ranked lists into one."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


def retrieve_context(
    query: str,
    history: list,
    chunks: list[dict],
    faiss_index: faiss.IndexFlatIP,
    embed_model: SentenceTransformer,
    bm25: BM25Okapi,
) -> tuple[str, bool]:
    """Hybrid retrieval: semantic + BM25, fused with RRF."""
    # Only user questions for search context (not assistant replies — avoids topic skew)
    recent_user_qs = " ".join(
        msg["content"] for msg in history[-4:]
        if msg.get("role") == "user" and isinstance(msg["content"], str)
    )
    search_text = (query + " " + recent_user_qs).strip()

    # Semantic search
    query_vec = embed_texts([search_text], embed_model).astype(np.float32)
    sem_scores, sem_indices = faiss_index.search(query_vec, TOP_K)
    semantic_ranked = [
        int(idx) for score, idx in zip(sem_scores[0], sem_indices[0])
        if score >= MIN_RETRIEVAL_SCORE and idx >= 0
    ]

    # BM25 keyword search
    tokens = list(jieba.cut(search_text))
    bm25_scores = bm25.get_scores(tokens)
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K]

    # Fuse with RRF and take top K
    fused = _rrf_combine([semantic_ranked, bm25_ranked])[:TOP_K]

    if not fused:
        return "", False

    parts = [f"[来源: {chunks[i]['source']}]\n{chunks[i]['text']}" for i in fused]
    return "\n\n---\n\n".join(parts), True


def augment_message(
    user_input: str,
    history: list,
    chunks: list[dict],
    faiss_index: faiss.IndexFlatIP,
    embed_model: SentenceTransformer,
    bm25: BM25Okapi,
) -> str:
    context, has_content = retrieve_context(user_input, history, chunks, faiss_index, embed_model, bm25)
    if has_content:
        return f"以下是老师材料中可能相关的内容：\n\n{context}\n\n---\n\n学生问题：{user_input}"
    else:
        return (
            "【系统提示：在知识库中未检索到与此问题相关的内容。"
            "请告知学生此问题超出你所掌握的老师材料范围。】\n\n"
            f"学生问题：{user_input}"
        )


# ── Setup ──────────────────────────────────────────────────────────────────────

def crawl_website(base_url: str, max_pages: int = 300) -> list[dict]:
    visited: set[str] = set()
    queue: list[str] = [base_url]
    page_texts: list[dict] = []
    base_domain = urlparse(base_url).netloc

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (compatible; TeacherAgentBot/1.0)"

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            print(f"  ⚠️  无法访问 {url}: {e}")
            continue

        if "html" not in resp.headers.get("content-type", "").lower():
            continue

        soup = BeautifulSoup(resp.text, "html.parser")

        # Respect <base href="..."> for resolving relative links
        base_tag = soup.find("base", href=True)
        link_base = base_tag["href"] if base_tag else url

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url
        text = soup.get_text(separator="\n", strip=True)
        if text:
            page_texts.append({"url": url, "title": title, "text": text})
            print(f"  🌐 已抓取: {title[:60]}")

        for a in soup.find_all("a", href=True):
            full_url = urljoin(link_base, a["href"]).split("#")[0]
            parsed = urlparse(full_url)
            if parsed.netloc != base_domain:
                continue
            if full_url not in visited and full_url not in queue:
                queue.append(full_url)

    return page_texts


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as e:
        print(f"  ⚠️  无法提取 {pdf_path.name}: {e}")
        return ""


def load_knowledge_base() -> tuple[dict, np.ndarray]:
    """Load chunks + embeddings from cache. Exits if missing."""
    if not os.path.exists(FILES_DB) or not os.path.exists(EMBEDDINGS_DB):
        print(f"❌ 知识库文件不存在，请先运行：python agent.py --build")
        raise SystemExit(1)

    with open(FILES_DB, encoding="utf-8") as f:
        state = json.load(f)

    if "chunks" not in state:
        print("❌ 知识库格式异常，请重新运行：python agent.py --build")
        raise SystemExit(1)

    embeddings = np.load(EMBEDDINGS_DB)
    print(f"📂 已加载知识库：{len(state['chunks'])} 个文本块，{embeddings.shape} 向量")
    return state, embeddings


def build_knowledge_base() -> None:
    """Crawl websites, extract PDFs, embed chunks, save to cache."""
    # 1. Crawl
    page_texts = []
    for url in WEBSITE_URLS:
        print(f"🔍 正在爬取网站: {url}")
        pages = crawl_website(url)
        print(f"   共找到 {len(pages)} 个网页")
        page_texts.extend(pages)
    print(f"\n   网站合计: {len(page_texts)} 个网页")

    # 2. Extract PDFs
    print(f"\n📚 正在提取 {PDF_DIR}/ 中的PDF文本…")
    pdf_texts = []
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        print(f"  📄 提取: {pdf_path.name}")
        text = extract_pdf_text(pdf_path)
        if text:
            pdf_texts.append({"filename": pdf_path.name, "text": text})
            print(f"     ✅ 提取了 {len(text):,} 字符")
        else:
            print(f"     ⚠️  未能提取文本（可能是扫描版PDF）")

    # 3. Chunk
    print(f"\n✂️  正在切分文本…")
    chunks = build_chunks(pdf_texts, page_texts)
    print(f"   共切分为 {len(chunks)} 个文本块")

    # 4. Embed
    print(f"\n🧠 正在生成语义向量…")
    model = load_embed_model()
    texts = [c["text"] for c in chunks]
    print(f"   正在 embedding {len(texts)} 个文本块（首次较慢）…", end="", flush=True)
    embeddings = embed_texts(texts, model)
    print(f" 完成，向量维度: {embeddings.shape}")

    # 5. Save
    state = {
        "teacher_name": TEACHER_NAME,
        "website_urls": WEBSITE_URLS,
        "chunks": chunks,
    }
    with open(FILES_DB, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    np.save(EMBEDDINGS_DB, embeddings)
    print(f"\n✨ 知识库构建完成 → {FILES_DB} + {EMBEDDINGS_DB}")


# ── System prompt ──────────────────────────────────────────────────────────────

def build_system_instruction(state: dict, feedback_summary: str = "") -> str:
    teacher_name = state.get("teacher_name", TEACHER_NAME)
    base = f"""你是{teacher_name}老师的AI助教，熟悉老师的整套教学体系，帮助学生理解和消化老师的教学内容。

【重要：消息格式说明】
每条用户消息会包含系统自动注入的「老师材料」背景片段，供你参考。这些材料是后台检索系统提供的，不是学生发送的——不要提及你看到了材料或文档，直接基于材料内容自然地回答学生的问题即可。

【核心原则】
1. 以提供的「老师材料」为依据来回答问题。用自己的语言自然地讲解、归纳和总结，不需要逐字引用，但内容必须忠实于老师的原意。
2. 如果材料中确实找不到相关内容，直接告诉学生："这个问题在老师现有的材料中没有找到，建议直接向老师请教。" 不要编造。
3. 来源引用灵活处理：当内容来自某本书或某篇文章时，自然地提一下出处即可（如"老师在意识强度一书中提到……"），不必每句话都标注。
4. 讲解时可以有教学温度——适当引导学生思考、帮助他们建立概念之间的联系，但不做主观价值判断，不替学生下结论。
5. 回答格式因问题而异：概念性问题（"什么是X""X是如何运作的"）用清晰的标题和要点组织，便于扫读；简单的追问或对话式问题直接自然作答，不要强行加标题。如果材料中有具体数字、比例或规模描述，务必在回答中提及，这些数据是老师体系中重要的参考依据。
6. 严禁混入任何宗教内容、宗教术语或宗教类比（佛教、道教、基督教、伊斯兰教等），老师的体系是独立的，不与任何宗教挂钩。
7. 不要把其他思想体系、哲学流派或心理学理论混入老师的观点中，保持老师体系的纯粹性。
8. 引用数字时必须明确该数字对应的具体指标——老师材料中有大量针对不同子检测点的量化数据（如各区检测点的百分比），这些是特定子指标的数值，不能混用或张冠李戴。例如：某个子检测点的"人类平均40%"不等于"人类平均意识强度为40%"；意识强度的整体等级用"级"衡量，人类平均约为3级。"""

    if feedback_summary:
        base += f"\n\n【根据用户反馈的改进要求】\n{feedback_summary}"

    return base


# ── Feedback system ────────────────────────────────────────────────────────────

def log_feedback(question: str, answer: str, rating: str, comment: str = ""):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment,
    }
    with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def collect_feedback(question: str, answer: str):
    try:
        raw = input("  [反馈] 这个回答准确吗？(回车跳过 / 输入 好 或 差 + 可选说明): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if not raw:
        return
    if raw.startswith("好") or raw.startswith("g"):
        log_feedback(question, answer, "good", raw[1:].strip())
        print("  ✅ 感谢反馈！\n")
    elif raw.startswith("差") or raw.startswith("b"):
        log_feedback(question, answer, "bad", raw[1:].strip())
        print("  📝 已记录，会帮助改进！\n")
    else:
        log_feedback(question, answer, "bad", raw)
        print("  📝 已记录！\n")


def load_feedback_summary(client: anthropic.Anthropic) -> str:
    if not os.path.exists(FEEDBACK_LOG):
        return ""
    entries = []
    with open(FEEDBACK_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    bad = [e for e in entries if e.get("rating") == "bad"]
    if len(bad) < 3:
        return ""

    recent_bad = bad[-20:]
    examples = "\n".join(
        f"- 问题：{e['question'][:80]} | 用户说：{e.get('comment', '无说明')[:60]}"
        for e in recent_bad
    )
    prompt = f"""以下是AI助教最近收到的负面反馈（共{len(recent_bad)}条）：

{examples}

请总结出2-3条具体的行为改进要求，用于指导AI助教改正，每条不超过30字。直接输出改进要求列表，不要多余解释。"""

    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception:
        return ""


# ── Chat loop ──────────────────────────────────────────────────────────────────

def chat():
    state, embeddings = load_knowledge_base()
    chunks = state["chunks"]

    embed_model = load_embed_model()
    print("   正在构建向量索引…", end="", flush=True)
    faiss_index = build_faiss_index(embeddings)
    print(f" 完成（{faiss_index.ntotal} 个向量）")
    bm25 = build_bm25_index(chunks)

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    print("🔍 正在分析历史反馈…", end="", flush=True)
    feedback_summary = load_feedback_summary(client)
    print(" 完成" if feedback_summary else " 暂无足够反馈数据")
    system_instruction = build_system_instruction(state, feedback_summary)

    teacher_name = state.get("teacher_name", TEACHER_NAME)
    print(f"\n{'='*50}")
    print(f"  {teacher_name} (AI助教 · 混合检索)")
    print(f"  已加载 {len(chunks)} 个知识块")
    print(f"  输入 'quit' 或 '退出' 结束对话")
    print(f"{'='*50}\n")

    history = []

    while True:
        try:
            user_input = input("学生: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "退出", "再见"):
            print("再见！")
            break

        augmented = augment_message(user_input, history, chunks, faiss_index, embed_model, bm25)
        trimmed_history = history[-(MAX_HISTORY_TURNS * 2):]
        api_messages = trimmed_history + [{"role": "user", "content": augmented}]

        print("助教: ", end="", flush=True)
        response_text = ""
        try:
            with client.messages.stream(
                model=MODEL,
                max_tokens=2048,
                system=system_instruction,
                messages=api_messages,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    response_text += text
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            continue

        print("\n")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})

        collect_feedback(user_input, response_text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_knowledge_base()
    else:
        chat()
