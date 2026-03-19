"""
agent.py — Teacher AI agent for 元吾氏.

On first run, uploads PDFs from ./files_to_feed/ and crawls
https://awakenology.org/Chinese/ to build the knowledge base.
State is cached to teacher_files.json for subsequent runs.

Usage:
    python agent.py
"""

import json
import os
import anthropic
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path

FILES_DB = "teacher_files.json"
PDF_DIR = Path("files_to_feed")
WEBSITE_URL = "https://awakenology.org/Chinese/"
TEACHER_NAME = "元吾氏"

MAX_PAGE_TEXT_CHARS = 3000
MAX_PAGES_IN_CONTEXT = 30


def crawl_website(base_url: str, max_pages: int = 100) -> list[dict]:
    """BFS-crawl the website. Returns list of {url, title, text}."""
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


def upload_local_pdfs(client: anthropic.Anthropic, existing_file_ids: dict) -> dict:
    """Upload PDFs from PDF_DIR to Files API, skipping already-uploaded ones."""
    file_ids = dict(existing_file_ids)
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        key = str(pdf_path)
        if key in file_ids:
            print(f"  ✅ 已上传 (跳过): {pdf_path.name}")
            continue
        print(f"  📤 上传: {pdf_path.name}")
        try:
            with open(pdf_path, "rb") as f:
                result = client.beta.files.upload(
                    file=(pdf_path.name, f, "application/pdf"),
                )
            file_ids[key] = {"file_id": result.id, "filename": pdf_path.name}
            print(f"     ✅ file_id: {result.id}")
        except Exception as e:
            print(f"  ⚠️  上传失败 {pdf_path.name}: {e}")
    return file_ids


def setup(client: anthropic.Anthropic) -> dict:
    """Build and cache the knowledge base. Reads from cache if available."""
    if os.path.exists(FILES_DB):
        with open(FILES_DB, encoding="utf-8") as f:
            state = json.load(f)
        print(f"📂 加载已有数据: {len(state.get('file_ids', {}))} 个已上传文件")
        return state

    print(f"🔍 正在爬取网站: {WEBSITE_URL}")
    page_texts = crawl_website(WEBSITE_URL)
    print(f"   共找到 {len(page_texts)} 个网页")

    print(f"\n📚 正在上传 {PDF_DIR}/ 中的PDF…")
    file_ids = upload_local_pdfs(client, {})

    state = {
        "teacher_name": TEACHER_NAME,
        "website_url": WEBSITE_URL,
        "page_texts": page_texts,
        "file_ids": file_ids,
    }

    with open(FILES_DB, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"\n✨ 知识库构建完成，已保存至 {FILES_DB}")
    return state


def build_system_prompt(state: dict) -> list[dict]:
    teacher_name = state.get("teacher_name", TEACHER_NAME)
    website_url = state.get("website_url", WEBSITE_URL)
    page_texts: list[dict] = state.get("page_texts", [])

    intro = f"""你就是{teacher_name}老师本人。学生正在向你请教问题，你要完全以老师本人的身份、口吻和视角来回答，就像在课堂上或办公室里亲自辅导学生一样。

角色要求：
- 用第一人称"我"来表达，比如"我在课上讲过……""我的意思是……""你可以翻翻我给你们的讲义第X页"
- 保持老师本人的语气和风格，不要表现出你是AI
- 根据课程材料（PDF讲义、课程网站）来回答，这些都是你自己准备的教学内容
- 如果学生问到材料之外的内容，以老师的口吻引导或给出你的判断
- 优先用中文回答，除非学生用其他语言提问
- 可以适当反问学生，帮助他们自己思考，而不只是直接给答案
- 不要说"根据材料""文档中提到"这类话——那是你自己写的内容，直接讲就好"""

    blocks: list[dict] = [{"type": "text", "text": intro}]

    if page_texts:
        pages_block = f"\n\n## 课程网站内容（来源: {website_url}）\n\n"
        for page in page_texts[:MAX_PAGES_IN_CONTEXT]:
            title = page.get("title", page["url"])
            text = page["text"][:MAX_PAGE_TEXT_CHARS]
            pages_block += f"### {title}\n{text}\n\n"

        blocks.append({
            "type": "text",
            "text": pages_block,
            "cache_control": {"type": "ephemeral"},
        })

    return blocks


def build_initial_doc_blocks(state: dict) -> list[dict]:
    """One document block per PDF, injected into the first user turn only."""
    return [
        {
            "type": "document",
            "source": {"type": "file", "file_id": info["file_id"]},
            "title": info["filename"],
        }
        for info in state.get("file_ids", {}).values()
    ]


def chat():
    client = anthropic.Anthropic()
    state = setup(client)

    teacher_name = state.get("teacher_name", TEACHER_NAME)
    pdf_count = len(state.get("file_ids", {}))
    page_count = len(state.get("page_texts", []))

    print(f"\n{'='*50}")
    print(f"  {teacher_name}老师 (AI助教)")
    print(f"  已加载 {pdf_count} 份讲义 + {page_count} 个课程网页")
    print(f"  输入 'quit' 或 '退出' 结束对话")
    print(f"{'='*50}\n")

    system_blocks = build_system_prompt(state)
    doc_blocks = build_initial_doc_blocks(state)
    messages: list[dict] = []

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

        # First turn: prepend PDF document blocks; they stay in history afterward
        if not messages and doc_blocks:
            content = doc_blocks + [{"type": "text", "text": user_input}]
        else:
            content = user_input

        messages.append({"role": "user", "content": content})

        print("助教: ", end="", flush=True)
        response_text = ""

        try:
            with client.beta.messages.stream(
                model="claude-opus-4-6",
                max_tokens=2048,
                system=system_blocks,
                messages=messages,
                betas=["files-api-2025-04-14"],
            ) as stream:
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                    response_text += text
        except anthropic.BadRequestError as e:
            print(f"\n❌ 请求错误: {e.message}")
            messages.pop()
            continue
        except anthropic.RateLimitError:
            print("\n⚠️  请求太频繁，请稍后再试")
            messages.pop()
            continue

        print("\n")
        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    chat()
