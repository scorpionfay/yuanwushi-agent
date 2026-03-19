# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and Running

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# Start the agent (auto-builds knowledge base on first run)
python agent.py
```

No build, test, or lint tooling exists in this project.

## Architecture

`agent.py` is the single entry point. On first run it builds the knowledge base automatically; on subsequent runs it loads from the cache.

### Knowledge base setup (first run)
1. BFS-crawls `https://awakenology.org/Chinese/` (same-domain only, up to 100 pages), extracting page text.
2. Uploads all PDFs from `./files_to_feed/` to the Anthropic Files API.
3. Saves everything to `teacher_files.json` — delete this file to force a rebuild.

### Conversation loop
Runs an interactive multi-turn chat using `claude-opus-4-6`. Key mechanics:
- **System prompt**: Teacher role instructions + up to 30 pages of website text (capped at 3000 chars/page). The website-text block uses `cache_control: ephemeral` for prompt caching.
- **PDF injection**: On the first user turn only, all uploaded PDFs are prepended as document blocks (via `file_id` references, `files-api-2025-04-14` beta). They persist in conversation history for subsequent turns.
- **Streaming**: Responses stream to stdout. Max 2048 tokens per response.
- Exit commands: `quit`, `exit`, `退出`

### State file format (`teacher_files.json`)
```json
{
  "teacher_name": "元吾氏",
  "website_url": "https://awakenology.org/Chinese/",
  "page_texts": [{"url": "...", "title": "...", "text": "..."}],
  "file_ids": {
    "files_to_feed/filename.pdf": {"file_id": "...", "filename": "..."}
  }
}
```
