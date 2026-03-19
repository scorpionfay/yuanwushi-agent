"""
agent.py — Teacher AI agent for 元吾氏 (powered by Google Gemini).

On first run, extracts text from PDFs in ./files_to_feed/ and crawls
https://awakenology.org/Chinese/ to build the knowledge base.
State is cached to teacher_files.json for subsequent runs.

Usage:
    python agent.py
"""

import json
import os
import google.generativeai as genai
import requests
import pypdf
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path

FILES_DB = "teacher_files.json"
PDF_DIR = Path("files_to_feed")
WEBSITE_URL = "https://awakenology.org/Chinese/"
TEACHER_NAME = "元吾氏"

MAX_PAGE_TEXT_CHARS = 3000
MAX_PAGES_IN_CONTEXT = 30
MAX_PDF_TEXT_CHARS = 100_000  # per PDF, ~25K tokens each


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


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file using pypdf."""
    try:
        reader = pypdf.PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as e:
        print(f"  ⚠️  无法提取 {pdf_path.name}: {e}")
        return ""


def setup() -> dict:
    """Build and cache the knowledge base. Reads from cache if available."""
    if os.path.exists(FILES_DB):
        with open(FILES_DB, encoding="utf-8") as f:
            state = json.load(f)
        print(f"📂 加载已有数据: {len(state.get('pdf_texts', []))} 份讲义 + {len(state.get('page_texts', []))} 个网页")
        return state

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

    state = {
        "teacher_name": TEACHER_NAME,
        "website_url": WEBSITE_URL,
        "page_texts": page_texts,
        "pdf_texts": pdf_texts,
    }

    with open(FILES_DB, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    print(f"\n✨ 知识库构建完成，已保存至 {FILES_DB}")
    return state


def build_system_instruction(state: dict) -> str:
    teacher_name = state.get("teacher_name", TEACHER_NAME)
    website_url = state.get("website_url", "")
    page_texts: list[dict] = state.get("page_texts", [])
    pdf_texts: list[dict] = state.get("pdf_texts", [])

    text = f"""你就是{teacher_name}老师本人。学生正在向你请教问题，你要完全以老师本人的身份、口吻和视角来回答，就像在课堂上或办公室里亲自辅导学生一样。

角色要求：
- 用第一人称"我"来表达，比如"我在课上讲过……""我的意思是……""你可以翻翻我给你们的讲义第X页"
- 保持老师本人的语气和风格，不要表现出你是AI
- 根据课程材料（PDF讲义、课程网站）来回答，这些都是你自己准备的教学内容
- 如果学生问到材料之外的内容，以老师的口吻引导或给出你的判断
- 优先用中文回答，除非学生用其他语言提问
- 可以适当反问学生，帮助他们自己思考，而不只是直接给答案
- 不要说"根据材料""文档中提到"这类话——那是你自己写的内容，直接讲就好"""

    if pdf_texts:
        text += "\n\n## 课程讲义内容\n\n"
        for pdf in pdf_texts:
            text += f"### {pdf['filename']}\n{pdf['text'][:MAX_PDF_TEXT_CHARS]}\n\n"

    if page_texts:
        text += f"\n\n## 课程网站内容（来源: {website_url}）\n\n"
        for page in page_texts[:MAX_PAGES_IN_CONTEXT]:
            title = page.get("title", page["url"])
            content = page["text"][:MAX_PAGE_TEXT_CHARS]
            text += f"### {title}\n{content}\n\n"

    return text


def chat():
    state = setup()

    api_key = os.environ.get("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    system_instruction = build_system_instruction(state)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_instruction,
    )

    teacher_name = state.get("teacher_name", TEACHER_NAME)
    pdf_count = len(state.get("pdf_texts", []))
    page_count = len(state.get("page_texts", []))

    print(f"\n{'='*50}")
    print(f"  {teacher_name}老师 (AI助教)")
    print(f"  已加载 {pdf_count} 份讲义 + {page_count} 个课程网页")
    print(f"  输入 'quit' 或 '退出' 结束对话")
    print(f"{'='*50}\n")

    chat_session = model.start_chat()

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

        print("助教: ", end="", flush=True)
        try:
            response = chat_session.send_message(user_input, stream=True)
            for chunk in response:
                if chunk.text:
                    print(chunk.text, end="", flush=True)
        except Exception as e:
            print(f"\n❌ 出错了: {e}")

        print("\n")


if __name__ == "__main__":
    chat()
