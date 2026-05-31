# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ContextCore** — an 8-hour hackathon prototype (W&B hackathon). A multi-agent middleware layer that intercepts LLM chat messages, manages persistent memory via a vector store, and prevents hallucination through a grounding agent.

**Demo goal:** A 50-turn conversation where ContextCore recalls facts from turn 2 at turn 48, while a baseline (no middleware) fails. The "wow moment" is turns 16, 20, and 25 in the built-in demo script.

---

## Current Status

**All code is complete and working.** Every file has been implemented, spec-reviewed, and quality-reviewed. The SSL issue on this machine (`UNABLE_TO_VERIFY_LEAF_SIGNATURE`) was resolved externally — do not re-add `NODE_TLS_REJECT_UNAUTHORIZED=0`.

---

## Setup & Run

```bash
npm install          # already done, node_modules exists
npm start            # starts server at http://localhost:3000
npm run dev          # same but with --watch for auto-restart
```

W&B eval scripts (run while server is running):
```bash
pip install -r eval/requirements.txt
python eval/eval.py             # memory quality eval — the main demo visual
python eval/retrieval_tune.py   # retrieval parameter sweep
```

---

## Models (NVIDIA NIM — free tier)

| Purpose | Model |
|---------|-------|
| All chat/LLM calls | `google/gemma-3n-e4b-it` |
| Embeddings (vector store) | `nvidia/nv-embed-v1` |

All calls go to `https://integrate.api.nvidia.com/v1` using the OpenAI SDK. API key is in `.env` as `NVIDIA_API_KEY`.

**Why these models:** The machine couldn't reach NVIDIA's API initially due to SSL issues. After resolving that, `google/gemma-3n-e4b-it` was confirmed working for chat and `nvidia/nv-embed-v1` for embeddings. All 5 original model references (Haiku/Sonnet/Nemotron) were replaced with these two.

---

## Architecture

```
User Message
     │
     ▼
[1. Intake Agent]         src/agents/intake.js       → gemma-3n-e4b-it (JSON extraction)
     │
     ▼
[2. Memory Read Agent]    src/agents/memoryRead.js   → pure cosine similarity, score > 0.6
     │
     ▼
[3. Context Packer]       src/agents/contextPacker.js → pure logic, no LLM
     │
     ▼
[4. Main LLM Call]        src/pipeline.js            → gemma-3n-e4b-it
     │
     ▼
[5. Grounding Agent]      src/agents/grounding.js    → gemma-3n-e4b-it (fact-check JSON)
     │
     ▼
[6. Memory Write]         src/agents/memoryWrite.js  → gemma-3n-e4b-it (async, non-blocking)
     │
     ▼
Response to User
```

---

## File Map

```
src/
├── server.js              Express server — POST /chat, GET /memories, DELETE /memories, POST /retrieve
├── pipeline.js            Wires all 5 agents; includes per-step timing (intake_ms, memRead_ms, llm_ms, ground_ms)
├── agents/
│   ├── intake.js          Classifies intent + extracts entities → { intent, entities, hasFact }
│   ├── memoryRead.js      Retrieves top-K memories with score > 0.6
│   ├── contextPacker.js   Assembles system prompt from memories + last 6 turns (pure logic)
│   ├── grounding.js       Validates LLM response against stored facts; corrects if contradiction found
│   └── memoryWrite.js     Extracts 2–5 atomic facts and stores them (called async, never blocks)
└── store/
    └── vectorStore.js     vectra-backed vector store; createStore() factory; embeddings via nvidia/nv-embed-v1

ui/
└── index.html             Full chat UI — memory inspector, A/B toggle, Run Demo button, contradiction banners

eval/
├── eval.py                W&B Option 2: runs 25-turn demo, scores ContextCore vs baseline accuracy
├── retrieval_tune.py      W&B Option 3: sweeps topK + scoreThreshold, logs recall to W&B
└── requirements.txt       wandb, weave, requests
```

---

## Server API

| Method | Path | Body / Params | Returns |
|--------|------|---------------|---------|
| POST | `/chat` | `{ message, sessionId?, useContextCore? }` | `{ response, groundingPassed, contradictions, memoriesUsed, tokenEstimate, intake, timing, turnCount }` |
| GET | `/memories` | — | Array of all stored fact metadata |
| DELETE | `/memories` | — | `{ ok: true }` — wipes `./vector-store` directory |
| POST | `/retrieve` | `{ query, topK?, scoreThreshold? }` | `{ results, topK, scoreThreshold }` — used by retrieval_tune.py |

**Baseline mode:** `useContextCore: false` — calls the LLM directly with last 5 turns, skips all agents.

---

## Key Implementation Details

- **Memory threshold:** facts filtered at `score > 0.6` in `memoryRead.js`
- **History window:** last 6 turns (`sessionHistory.slice(-6)`) passed to context packer
- **Memory write is async:** `runMemoryWrite(...).catch(console.error)` — never blocks the response
- **Grounding short-circuit:** if no memories exist yet, grounding is skipped entirely
- **All JSON from LLM** is defensively parsed — markdown fences stripped, errors return safe defaults
- **Sessions** are in-memory in `server.js` (keyed by `sessionId`), lost on server restart
- **Vector store** is file-backed at `./vector-store/` (gitignored); `DELETE /memories` wipes it
- **vectorStore** uses singleton `indexReadyPromise` to prevent race conditions on init
- **`/retrieve` endpoint** exists specifically for `retrieval_tune.py` — passes `topK` and `scoreThreshold` directly to the store

---

## UI Features

- **ContextCore ON/OFF toggle** — switches between pipeline and baseline mode
- **A/B Mode toggle** — sends each message to both modes simultaneously, shows side-by-side responses (green border = ContextCore, grey = baseline)
- **Memory Inspector sidebar** — live list of stored facts; stars (★) highlight which facts were used in the last turn
- **Run Demo button** — auto-plays the full 25-turn demo script with 1500ms delays; turns 16, 20, 25 shown in A/B mode
- **Pipeline status cycling** — while waiting, cycles through "Analyzing...", "Retrieving memories...", "Generating response...", "Checking for contradictions..." every 800ms
- **Contradiction warning banner** — shown above response when `groundingPassed === false`
- **Rate limit handling** — shows friendly message instead of raw API error on 429
- **Reset button** — calls `DELETE /memories`, clears chat

---

## Demo Script (built into UI "Run Demo" button)

```
Turns 1–5:   Setup (name, DB=PostgreSQL+Redis, language=TypeScript, bug fix, fintech customer)
Turns 6–15:  Filler (unrelated questions to push facts out of context window)
Turn 16:     "What database are we using?" → A/B: ContextCore recalls, baseline forgets
Turn 20:     "What language does my team use?" → A/B: same contrast
Turn 25:     "Actually we switched to MongoDB last month." → Grounding catches contradiction
```

---

## W&B Integration

`eval/eval.py` runs the 25-turn demo programmatically via the `/chat` API, then scores 4 recall questions:
- "What database does the user use?" → expected: postgresql
- "What programming language does the team use?" → expected: typescript
- "What did the user fix last week?" → expected: race condition
- "What kind of company is their biggest customer?" → expected: fintech

Logs per-question pass/fail + final accuracy to W&B project `contextcore-demo`. The accuracy comparison chart (ContextCore vs baseline) is the main demo visual.

`eval/retrieval_tune.py` sweeps topK ∈ [3,5,8] × scoreThreshold ∈ [0.5,0.6,0.65,0.75,0.85] and logs recall to W&B project `contextcore-retrieval-tuning`.
