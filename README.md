# ContextCore

> **"We Built an AI That Never Forgets (vs One That Does)"**

ContextCore is a multi-agent middleware layer that gives LLMs persistent, long-term memory — without relying on conversation history. Every fact a user shares is extracted, embedded, and stored in a vector database. On the next question, only the relevant facts are retrieved and injected into the prompt.

The result: ContextCore can recall something you said at turn 1 when you ask about it at turn 50. A plain baseline with a sliding context window cannot.

---

## The Problem

Every LLM API call is stateless. To maintain a conversation you pass the message history back on every request. This works for a few turns, but breaks down in two ways:

- **Context window overflow** — models have a hard token limit. Once history exceeds it, you drop old messages. Facts from early in the conversation are silently forgotten.
- **Noise** — passing raw history sends everything to the model regardless of relevance, wasting tokens and degrading response quality.

The standard fix — a **sliding window** — keeps only the last N messages. Simple, but it means any fact outside that window is permanently gone.

---

## The Solution

ContextCore replaces the sliding window with a 7-agent pipeline:

```
User Message
     │
     ▼
┌─────────────────────────┐
│  1. Understanding        │  Is retrieval needed? Expand the search query.
├─────────────────────────┤
│  2. Memory Read          │  Vector similarity search → top-K candidates
├─────────────────────────┤
│  3. Relevance Decision   │  LLM filters to only the facts that matter
├─────────────────────────┤
│  4. Context Packer       │  Builds the final prompt: facts + current message
├─────────────────────────┤
│  5. Main LLM             │  Generates the response
├─────────────────────────┤
│  6. Grounding Agent      │  Catches hallucinations before they reach the user
├─────────────────────────┤
│  7. Memory Write (async) │  Extracts 2–5 facts and stores them for the future
└─────────────────────────┘
```

The main LLM never sees raw conversation history. It sees only **retrieved facts** + the **current question**. Memory scales indefinitely.

---

## Demo

The built-in **Run Demo** button plays a 12-turn script:

| Turns | What happens |
|-------|-------------|
| 1–5   | User shares facts: name, database stack, language, a bug fix, their customer |
| 6–9   | Unrelated filler questions push those facts outside the baseline's 3-turn window |
| 10    | "What database are we using?" — ContextCore recalls it, baseline forgets |
| 11    | "What language does my team write in?" — same contrast |
| 12    | "Actually we switched to MongoDB" — Grounding agent catches the contradiction |

The UI shows both responses side-by-side in real time.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Server | Node.js + Express |
| LLM | `OpenPipe/Qwen3-14B-Instruct` via [WandB Serverless Inference](https://wandb.ai/inference) |
| Vector Store | [Vectra](https://github.com/Stevenic/vectra) (local file-backed) |
| Embeddings | Local DJB2 feature-hashing (512-dim) — no embedding API needed |
| Eval | Python + W&B for logging and comparison charts |
| UI | Vanilla HTML/CSS/JS — no framework |

---

## Quickstart

### Prerequisites

- Node.js 18+
- A [WandB](https://wandb.ai) account with an API key

### 1. Install dependencies

```bash
cd Hackathon-WnB
npm install
```

### 2. Set your API key

Create a `.env` file:

```
WANDB_API_KEY=your_key_here
```

### 3. Start the server

```bash
node --require ./setup.cjs src/server.js
```

Or for development with auto-restart:

```bash
node --watch --require ./setup.cjs src/server.js
```

### 4. Open the UI

Navigate to `http://localhost:3000` in your browser.

---

## Project Structure

```
Hackathon-WnB/
├── src/
│   ├── server.js                 Express server — /chat, /memories, /retrieve
│   ├── pipeline.js               Orchestrates the 7-agent pipeline
│   ├── agents/
│   │   ├── understanding.js      Agent 1 — intent classification + query expansion
│   │   ├── memoryRead.js         Agent 2 — vector similarity retrieval
│   │   ├── relevanceDecision.js  Agent 3 — LLM-based relevance filter
│   │   ├── contextPacker.js      Agent 4 — prompt assembly (no LLM call)
│   │   ├── grounding.js          Agent 6 — hallucination/contradiction detection
│   │   └── memoryWrite.js        Agent 7 — atomic fact extraction + storage
│   └── store/
│       └── vectorStore.js        Local hash embeddings + Vectra index
├── ui/
│   └── index.html                Full chat UI — A/B mode, memory inspector, timeline
├── eval/
│   ├── test_context_overflow.py  A/B overflow test with W&B logging
│   └── requirements.txt
├── setup.cjs                     Node 18 globalThis.File polyfill
├── ARCHITECTURE.md               Deep-dive on every agent and design decision
└── .env                          WANDB_API_KEY (not committed)
```

---

## API Reference

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| `POST` | `/chat` | `{ message, sessionId?, useContextCore? }` | `{ response, memoriesUsed, groundingPassed, contradictions, tokenEstimate, timing, turnCount }` |
| `GET` | `/memories` | — | Array of all stored facts |
| `DELETE` | `/memories` | — | Wipes the vector store |
| `POST` | `/retrieve` | `{ query, topK?, scoreThreshold? }` | Raw similarity results — used by eval scripts |

Set `useContextCore: false` to get baseline mode (plain sliding window, no agents).

---

## UI Features

| Feature | Description |
|---------|-------------|
| **A/B mode** | Every message goes to both ContextCore and Baseline simultaneously — responses shown side-by-side |
| **Memory inspector** | Live sidebar showing every stored fact; facts used in the last turn are highlighted with ★ |
| **Context window bars** | Footer progress bars showing how full each mode's context window is, color-coded green → orange → red |
| **Memory callout** | Green badge on the ContextCore side showing exactly which facts were retrieved and their similarity score |
| **Window warning** | Orange warning on the Baseline side once turns fall outside the 3-turn window |
| **Turn timeline** | Dot row above the input: blue = in baseline window, grey = forgotten, green = accessible via ContextCore memory |
| **Memory hit rate** | Footer counter showing how many turns ContextCore successfully retrieved at least one fact |
| **Grounding banner** | Red warning shown when the grounding agent detects a contradiction, listing the specific conflict |
| **Run Demo** | One-click 12-turn scripted demo that shows the recall contrast and grounding in action |

---

## Eval: Overflow Test

Runs a rigorous A/B comparison and logs everything to W&B:

```bash
pip install -r eval/requirements.txt
python eval/test_context_overflow.py
```

The test:
1. Plants three anchor facts (a codeword, a project name, a lucky number) at turn 1
2. Fills 7 turns with unrelated questions — pushing the anchors out of the baseline window
3. Asks recall questions on all three facts
4. Scores each mode pass/fail and logs a comparison table to W&B

Expected result: Baseline scores 0%, ContextCore scores 100%.

---

## How the Vector Store Works

WandB Serverless Inference has no embedding endpoint. ContextCore uses a **local feature-hashing** approach instead:

1. Text is tokenized (lowercased, stop words removed)
2. Each word is hashed with DJB2 into a 512-dimensional bucket: `vec[hash(word) % 512] += 1`
3. Adjacent word pairs (bigrams) add half-weight: `vec[hash("w1_w2") % 512] += 0.5`
4. The vector is L2-normalized
5. Retrieval uses cosine similarity with a threshold of `0.2`

The Understanding agent compensates for the weaker embeddings by expanding queries with synonyms before hitting the store.

---

## Architecture Deep-Dive

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for a full breakdown of every agent, design decision, fail-safes, and a concrete data flow walkthrough.

---

## Comparison

| | Baseline | ContextCore |
|--|----------|-------------|
| History in prompt | Last 6 messages | None — retrieved facts only |
| Recall beyond 3 turns | No | Yes |
| Context window overflow | Silently forgets | Never — memory scales independently |
| Hallucination detection | None | Grounding agent corrects before delivery |
| LLM calls per turn | 1 | 4 sync + 1 async |
| Latency overhead | ~0ms | ~3–8s (WandB free tier) |
