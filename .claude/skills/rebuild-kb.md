# rebuild-kb

重建元吾氏 AI 助教的知识库。

运行以下命令重新爬取网站（中文 + 英文）、提取 PDF，并生成语义向量索引：

```
!python3 agent.py --build
```

完成后会生成两个文件：
- `teacher_files.json` — 文本块元数据
- `teacher_embeddings.npy` — 语义向量（FAISS 索引用）

**注意：这两个文件均已在 `.gitignore` 中排除，不要手动 `git add` 它们。**
