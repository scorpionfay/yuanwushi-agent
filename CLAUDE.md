# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and Running

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=your_key_here

# Build knowledge base (required before first run)
python3 agent.py --build

# Start the agent
python3 agent.py
```

No build, test, or lint tooling exists in this project.

## Architecture

`agent.py` is the single entry point with two modes:
- `python3 agent.py --build` — crawl websites + extract PDFs + embed all chunks → saves `teacher_files.json` + `teacher_embeddings.npy`
- `python3 agent.py` — load cached knowledge base and start chat (never triggers a crawl)

### Knowledge base build (`--build`)
1. BFS-crawls each URL in `WEBSITE_URLS` (respects `<base href>` tags, same-domain only, up to 300 pages each).
2. Extracts text from all PDFs in `./files_to_feed/`.
3. Chunks all text (600 chars, 100 overlap).
4. Embeds all chunks using `paraphrase-multilingual-MiniLM-L12-v2` (local model, supports Chinese + English).
5. Saves chunks to `teacher_files.json`, embeddings to `teacher_embeddings.npy`.

### Retrieval (semantic RAG)
- Each user query is embedded with the same model.
- FAISS `IndexFlatIP` (cosine similarity on normalised vectors) retrieves top-K chunks.
- Chunks below `MIN_RETRIEVAL_SCORE = 0.3` are discarded; if nothing passes threshold, the model is told no relevant content was found.
- Recent conversation history is appended to the query for better follow-up retrieval.

### Conversation loop
- Model: `claude-sonnet-4-6`, streaming, max 2048 tokens.
- System prompt enforces: stay grounded in retrieved material, no religious content, no subjective opinions, cite sources.
- Feedback collected after each response (`feedback_log.jsonl`); ≥3 bad entries → Claude analyses patterns and adds improvement rules to system prompt on next startup.
- Exit commands: `quit`, `exit`, `退出`

### Key files
| File | Purpose | Git |
|------|---------|-----|
| `teacher_files.json` | Chunk metadata cache | ignored |
| `teacher_embeddings.npy` | Embedding vectors | ignored |
| `feedback_log.jsonl` | User feedback log | ignored |
| `files_to_feed/` | Source PDFs | ignored |
