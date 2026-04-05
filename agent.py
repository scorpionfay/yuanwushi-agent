"""
agent.py — Teacher AI agent for 元吾氏 (powered by Claude + semantic RAG).

Usage:
    python agent.py           # start chat (requires pre-built knowledge base)
    python agent.py --build   # crawl websites + extract PDFs, build knowledge base
"""

import json
import os
import datetime
import sys
import anthropic
import requests
import pypdf
import numpy as np
import faiss
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
MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 8               # chunks to retrieve per query
MAX_HISTORY_TURNS = 10  # keep last N turns in API history
MIN_RETRIEVAL_SCORE = 0.3  # cosine similarity threshold (0-1); below this = not relevant


# ── Text processing ────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        if len(chunk.strip()) > 50:
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def build_chunks(pdf_texts: list[dict], page_texts: list[dict]) -> list[dict]:
    chunks = []
    for pdf in pdf_texts:
        chunks.extend(chunk_text(pdf["text"], pdf["filename"]))
    for page in page_texts:
        chunks.extend(chunk_text(page["text"], page.get("title", page["url"])))
    return chunks


# ── Semantic retrieval ─────────────────────────────────────────────────────────

def load_embed_model() -> SentenceTransformer:
    print(f"📡 正在加载语义检索模型（{EMBED_MODEL}）…", end="", flush=True)
    model = SentenceTransformer(EMBED_MODEL)
    print(" 完成")
    return model


def embed_texts(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine similarity on normalised vecs
    index.add(embeddings.astype(np.float32))
    return index


def retrieve_context(
    query: str,
    history: list,
    chunks: list[dict],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
) -> tuple[str, bool]:
    """Returns (context_string, has_relevant_content)."""
    # Only use recent USER questions for retrieval context (not assistant replies).
    # Including assistant replies in the search vector skews retrieval toward the
    # previous topic instead of the current question.
    recent_user_qs = " ".join(
        msg["content"] for msg in history[-4:]
        if msg.get("role") == "user" and isinstance(msg["content"], str)
    )
    search_text = (query + " " + recent_user_qs).strip()
    query_vec = embed_texts([search_text], model).astype(np.float32)

    scores, indices = index.search(query_vec, TOP_K)
    scores, indices = scores[0], indices[0]

    relevant = [
        (score, idx)
        for score, idx in zip(scores, indices)
        if score >= MIN_RETRIEVAL_SCORE and idx >= 0
    ]
    if not relevant:
        return "", False

    parts = [f"[来源: {chunks[idx]['source']}]\n{chunks[idx]['text']}" for _, idx in relevant]
    return "\n\n---\n\n".join(parts), True


def augment_message(
    user_input: str,
    history: list,
    chunks: list[dict],
    index: faiss.IndexFlatIP,
    model: SentenceTransformer,
) -> str:
    context, has_content = retrieve_context(user_input, history, chunks, index, model)
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

【核心原则】
1. 以提供的「老师材料」为依据来回答问题。用自己的语言自然地讲解、归纳和总结，不需要逐字引用，但内容必须忠实于老师的原意。
2. 如果材料中确实找不到相关内容，直接告诉学生："这个问题在老师现有的材料中没有找到，建议直接向老师请教。" 不要编造。
3. 来源引用灵活处理：当内容来自某本书或某篇文章时，自然地提一下出处即可（如"老师在意识强度一书中提到……"），不必每句话都标注。
4. 讲解时可以有教学温度——适当引导学生思考、帮助他们建立概念之间的联系，但不做主观价值判断，不替学生下结论。
5. 严禁混入任何宗教内容、宗教术语或宗教类比（佛教、道教、基督教、伊斯兰教等），老师的体系是独立的，不与任何宗教挂钩。
6. 不要把其他思想体系、哲学流派或心理学理论混入老师的观点中，保持老师体系的纯粹性。"""

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
    index = build_faiss_index(embeddings)
    print(f" 完成（{index.ntotal} 个向量）")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    print("🔍 正在分析历史反馈…", end="", flush=True)
    feedback_summary = load_feedback_summary(client)
    print(" 完成" if feedback_summary else " 暂无足够反馈数据")
    system_instruction = build_system_instruction(state, feedback_summary)

    teacher_name = state.get("teacher_name", TEACHER_NAME)
    print(f"\n{'='*50}")
    print(f"  {teacher_name} (AI助教 · 语义检索)")
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

        augmented = augment_message(user_input, history, chunks, index, embed_model)
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
