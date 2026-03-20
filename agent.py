"""
agent.py — Teacher AI agent for 元吾氏 (powered by Google Gemini + BM25 RAG).

On first run, extracts text from PDFs in ./files_to_feed/ and crawls
https://awakenology.org/Chinese/, then chunks and indexes everything.
State is cached to teacher_files.json for subsequent runs.

Usage:
    python agent.py
"""

import json
import os
from google import genai
from google.genai import types
import requests
import pypdf
import jieba
from rank_bm25 import BM25Okapi
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path

FILES_DB = "teacher_files.json"
PDF_DIR = Path("files_to_feed")
WEBSITE_URL = "https://awakenology.org/Chinese/"
TEACHER_NAME = "元吾氏"
MODEL = "gemini-3.1-flash-lite-preview"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
TOP_K = 8  # chunks to retrieve per query
MAX_HISTORY_TURNS = 10  # keep last N turns in API history


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


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    print("   正在构建检索索引…", end="", flush=True)
    corpus = [list(jieba.cut(c["text"])) for c in chunks]
    index = BM25Okapi(corpus)
    print(f" 完成（{len(chunks)} 块）")
    return index


def retrieve_context(query: str, history: list, chunks: list[dict], bm25: BM25Okapi) -> str:
    # Combine current query with recent history for better follow-up retrieval
    recent = " ".join(
        part["text"]
        for msg in history[-4:]
        for part in msg.get("parts", [])
    )
    search_text = query + " " + recent
    query_tokens = list(jieba.cut(search_text))
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:TOP_K]
    parts = [f"[{chunks[i]['source']}]\n{chunks[i]['text']}" for i in top_indices]
    return "\n\n---\n\n".join(parts)


def augment_message(user_input: str, history: list, chunks: list[dict], bm25: BM25Okapi) -> str:
    context = retrieve_context(user_input, history, chunks, bm25)
    return f"以下是可能相关的参考内容：\n\n{context}\n\n---\n\n问题：{user_input}"


# ── Setup ──────────────────────────────────────────────────────────────────────

def crawl_website(base_url: str, max_pages: int = 100) -> list[dict]:
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

        content_type = resp.headers.get("content-type", "")
        if "html" not in content_type.lower():
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else url
        text = soup.get_text(separator="\n", strip=True)
        if text:
            page_texts.append({"url": url, "title": title, "text": text})
            print(f"  🌐 已抓取: {title[:60]}")

        for a in soup.find_all("a", href=True):
            full_url = urljoin(url, a["href"]).split("#")[0]
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


def setup() -> dict:
    """Build and cache the chunked knowledge base. Migrates old format if needed."""
    if os.path.exists(FILES_DB):
        with open(FILES_DB, encoding="utf-8") as f:
            state = json.load(f)

        # Migrate old format (pdf_texts/page_texts) to new chunked format
        if "chunks" not in state:
            print("📦 迁移到分块检索格式…")
            chunks = build_chunks(
                state.pop("pdf_texts", []),
                state.pop("page_texts", []),
            )
            state["chunks"] = chunks
            with open(FILES_DB, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            print(f"   ✅ 完成，共 {len(chunks)} 个文本块")
        else:
            print(f"📂 加载已有数据: {len(state['chunks'])} 个文本块")
        return state

    # Fresh setup
    print(f"🔍 正在爬取网站: {WEBSITE_URL}")
    page_texts = crawl_website(WEBSITE_URL)
    print(f"   共找到 {len(page_texts)} 个网页")

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

    print(f"\n✂️  正在切分文本…")
    chunks = build_chunks(pdf_texts, page_texts)
    print(f"   共切分为 {len(chunks)} 个文本块")

    state = {
        "teacher_name": TEACHER_NAME,
        "website_url": WEBSITE_URL,
        "chunks": chunks,
    }
    with open(FILES_DB, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"\n✨ 知识库构建完成，已保存至 {FILES_DB}")
    return state


# ── System prompt ──────────────────────────────────────────────────────────────

def build_system_instruction(state: dict) -> str:
    teacher_name = state.get("teacher_name", TEACHER_NAME)
    return f"你是{teacher_name}老师的AI助教，帮助学生理解{teacher_name}老师的教学内容。"


# ── Chat loop ──────────────────────────────────────────────────────────────────

def chat():
    state = setup()
    chunks = state["chunks"]
    bm25 = build_bm25_index(chunks)

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    system_instruction = build_system_instruction(state)

    teacher_name = state.get("teacher_name", TEACHER_NAME)
    print(f"\n{'='*50}")
    print(f"  {teacher_name} (AI助教)")
    print(f"  已加载 {len(chunks)} 个知识块，检索索引就绪")
    print(f"  输入 'quit' 或 '退出' 结束对话")
    print(f"{'='*50}\n")

    history = []  # stores original messages (without injected context)

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

        # Retrieve relevant context (query + recent history for follow-up awareness)
        augmented = augment_message(user_input, history, chunks, bm25)
        trimmed_history = history[-(MAX_HISTORY_TURNS * 2):]
        api_messages = trimmed_history + [{"role": "user", "parts": [{"text": augmented}]}]

        print("助教: ", end="", flush=True)
        response_text = ""
        try:
            stream = client.models.generate_content_stream(
                model=MODEL,
                contents=api_messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response_text += chunk.text
        except Exception as e:
            print(f"\n❌ 出错了: {e}")
            continue

        print("\n")
        # Store original (unaugmented) messages in history
        history.append({"role": "user", "parts": [{"text": user_input}]})
        history.append({"role": "model", "parts": [{"text": response_text}]})


if __name__ == "__main__":
    chat()
