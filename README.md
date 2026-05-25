# RAG Demo

Two standalone scripts that explains Retrieval-Augmented Generation (RAG) using OpenAI and FAISS.

| Script | What it explains |
|---|---|
| `simple_rag.py` | The full RAG loop: embed → store → retrieve → generate |
| `lyrics_rag.py` | Retrieval at scale: chunking, summaries, similarity thresholds, source comparison |

---

## What is RAG?

RAG is a pattern for grounding an LLM's answers in a specific set of documents:

1. **Embed** — convert text into vectors using an embedding model
2. **Store** — save the vectors in a vector database (here: FAISS)
3. **Retrieve** — for a question, find the most similar vectors
4. **Generate** — pass the retrieved text as context to an LLM

`simple_rag.py` shows all four steps. `lyrics_rag.py` focuses on steps 1–3 at scale with real song lyrics.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key

Create a `.env` file in the `rag/` directory:

```
OPENAI_API_KEY=sk-...
```

Both scripts load this file automatically.

---

## simple_rag.py

A minimal, fully commented RAG pipeline. The entire flow — embedding, storing, retrieving, and answering — runs in one script against five hardcoded facts about Zürich.

### What it does

1. Embeds short text documents using `text-embedding-3-small`
2. Stores the vectors in a local FAISS index (cached to disk)
3. Retrieves the top 2 most relevant documents for each question
4. Feeds the retrieved text into `gpt-4o-mini` (or other models) and prints the answer

### Run it

```bash
python simple_rag.py
```

### Expected output

```
Test question 1: What is the Roman name of Zurich?
Retrieved documents from FAISS:
- Zürich began as a Roman customs station called Turicum...
- Zürich is Switzerland's largest city...
Answer: The Roman name of Zurich is Turicum.
```

### Key concepts to notice

- **FAISS caching**: the index is saved to `embeddings/simple_rag/`. Delete it to force re-embedding.
- **`k=2`**: only the top 2 documents are passed to the model. Try changing it.
- **Prompt constraint**: `"Answer based only on the following context"` prevents the model from using its general knowledge.

---

## lyrics_rag.py

A retrieval-only demo using a CSV of song lyrics (`lyrics.csv`). It finds songs matching a topic by semantic similarity — no LLM generation step.

Supports three retrieval strategies and a compare mode to see them side by side.

### Retrieval strategies

| Strategy | What gets embedded | Good for |
|---|---|---|
| `lyrics` | Raw lyric chunks (700 chars each) | Literal phrase or word matches |
| `summaries` | LLM-generated song summaries | Thematic or emotional queries |
| `both` | Lyric chunks + summaries combined | Best overall coverage (default) |

### Run it — interactive mode

```bash
python lyrics_rag.py
```

Prompts you to type a topic and returns the top 3 matching songs. Type `exit` to quit.

### Run it — single topic

```bash
python lyrics_rag.py --topic "heartbreak"
```

### Choose a retrieval strategy

```bash
python lyrics_rag.py --retrieval-source lyrics
python lyrics_rag.py --retrieval-source summaries
python lyrics_rag.py --retrieval-source both
```

### Compare all three strategies side by side

```bash
python lyrics_rag.py --compare
python lyrics_rag.py --compare --topic "growing up"
```

Compare mode builds all three FAISS indexes and runs every query against all three, showing which songs each strategy finds and whether the match came from a lyric chunk or a summary.

### Expected output (compare mode)

```
Comparing retrieval sources for: heartbreak

-- Source: lyrics --
  1. Someone Like You by Adele (cosine similarity 0.61, matched via lyrics)
  2. ...

-- Source: summaries --
  1. Someone Like You by Adele (cosine similarity 0.74, matched via summary)
  2. ...

-- Source: both --
  1. Someone Like You by Adele (cosine similarity 0.74, matched via summary)
  2. ...
```

### Key concepts to notice

- **Chunking**: long lyrics are split into 700-character chunks. Run the script and look at the "Chunk examples" printed at startup — each chunk is what actually gets embedded.
- **Summaries vs. chunks**: summaries are semantically denser, so they often score higher on thematic queries. Lyric chunks can win on literal word matches.
- **Cosine similarity threshold**: lyric chunks below `0.25` similarity are filtered out. Summaries are not filtered because they're richer and low scores are still meaningful.
- **Caching**: FAISS indexes are saved to `embeddings/lyrics_rag/` and song summaries to `summaries/`. Both are keyed by content hash, so they rebuild automatically only when the data or models change.

### lyrics.csv format

The script expects a CSV with at least these columns:

| Column | Description |
|---|---|
| `song_title` | Song name |
| `primary_artist` | Artist name |
| `plain_lyrics` | Full song lyrics as plain text |
| `lyrics_found` | Boolean — `True` if lyrics are present |

---

## File structure

```
rag/
├── simple_rag.py             # Minimal RAG demo with generation
├── lyrics_rag.py             # Retrieval demo with lyrics at scale
├── lyrics.csv                # Song lyrics dataset
├── lyrics_with_summaries.csv # Generated by lyrics_rag.py when summaries are needed
├── requirements.txt
├── .env                      # Your OPENAI_API_KEY (not committed)
├── embeddings/
│   ├── simple_rag/           # Cached FAISS index for simple_rag.py
│   └── lyrics_rag/           # Cached FAISS indexes for lyrics_rag.py
└── summaries/                # Cached song summary JSON (lyrics_rag.py)
```

---

## Tuning parameters

Both scripts have constants at the top you can change to experiment.

### simple_rag.py

| Constant | Default | Effect |
|---|---|---|
| `EMBEDDING_MODEL_NAME` | `text-embedding-3-small` | Embedding model used |
| `CHAT_MODEL_NAME` | `gpt-4o-mini` | LLM used for answers |

### lyrics_rag.py

| Constant | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | `700` | Characters per lyric chunk — smaller = more precise, more chunks |
| `CHUNK_OVERLAP` | `0` | Overlap between chunks — increase to avoid cutting phrases mid-sentence |
| `TOP_K` | `4` | Candidates retrieved from FAISS before deduplication |
| `TOP_SONGS` | `3` | Final number of songs returned |
| `MIN_CHUNK_COSINE_SIMILARITY` | `0.25` | Lyric chunks below this score are ignored |
| `RETRIEVAL_SOURCE` | `"both"` | Default strategy when no flag is passed |
