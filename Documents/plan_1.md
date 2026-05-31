# ContextCore — Detailed Implementation Plan

**Based on:** `Documents/ContextCore_PRD.md`
**Status:** All source code needs to be built from scratch. Only the PRD and CLAUDE.md exist.
**Stack:** Node.js (ESM) + NVIDIA NIM (free tier) + vectra + Express + W&B (Python eval scripts)

---

## Prerequisites

1. **NVIDIA API Key** — sign in at https://build.nvidia.com → generate an API key
2. **Node.js 20+** installed
3. **Python 3.11+** installed (for W&B eval scripts only)
4. **W&B account** — sign up at https://wandb.ai, run `wandb login` once

---

## Phase 1 — Project Skeleton + Vector Store (Hr 0–1)

### Step 1.1: Initialize project

Create the directory structure:
```
contextcore/
├── src/
│   ├── agents/
│   │   ├── intake.js
│   │   ├── memoryWrite.js
│   │   ├── memoryRead.js
│   │   ├── grounding.js
│   │   └── contextPacker.js
│   ├── store/
│   │   └── vectorStore.js
│   ├── pipeline.js
│   └── server.js
├── ui/
│   └── index.html
├── eval/
│   ├── eval.py            (W&B Option 2 — memory quality)
│   ├── retrieval_tune.py  (W&B Option 3 — param sweep)
│   └── requirements.txt
├── .env                   (NVIDIA_API_KEY=... , gitignored)
├── .env.example
├── .gitignore
└── package.json
```

Files to create:
- **`package.json`** — type: "module", scripts: `start` and `dev` (with `--watch`), deps: `openai`, `express`, `cors`, `vectra`, `dotenv`
- **`.env.example`** — template with `NVIDIA_API_KEY=your_key_here` and `PORT=3000`
- **`.gitignore`** — `.env`, `node_modules/`, `vector-store/`

### Step 1.2: Build the vector store (`src/store/vectorStore.js`)

This is the memory backbone. Uses `vectra` (local, file-backed) with NVIDIA NIM embeddings.

**Exports a factory `createStore()`** returning an object with three methods:

| Method | Description |
|--------|-------------|
| `storeFact(fact, metadata)` | Embed the fact string via `nvidia/nv-embedqa-e5-v5`, insert into vectra index at `./vector-store` with metadata `{ text, ts, ...metadata }` |
| `retrieveFacts(query, topK=5)` | Embed the query, run `index.queryItems(vector, topK)`, return array of `{ text, score, ts, ...metadata }` |
| `getAllFacts()` | Return all items' metadata (for the memory inspector UI) |

**Implementation detail:**
- The OpenAI client is created once at module level, pointed at `https://integrate.api.nvidia.com/v1` with `process.env.NVIDIA_API_KEY`
- A helper `ensureIndex()` lazily creates the vectra index directory on first call
- Embedding model: `nvidia/nv-embedqa-e5-v5`, with `encoding_format: 'float'`
- Access the embedding vector at `res.data[0].embedding` (OpenAI-compatible response shape)

### Step 1.3: Manual verification

Write a small throwaway test at the bottom of `vectorStore.js` (or a separate `test-store.js`):
1. Call `storeFact("user prefers TypeScript")`
2. Call `retrieveFacts("what language does the user like?")`
3. Confirm it returns the fact with score > 0.7

**Done when:** Embedding round-trip works. Delete the test code before moving on.

---

## Phase 2 — Intake + Memory Write Agents (Hr 1–2)

### Step 2.1: Intake Agent (`src/agents/intake.js`)

**Purpose:** Classify intent and extract entities from each incoming user message.

**Function signature:** `runIntake(message, client)` → `{ intent, entities, hasFact }`

**Implementation:**
- Model: `meta/llama-3.1-8b-instruct` (fast, cheap — this is a simple extraction task)
- System prompt instructs the model to return **JSON only**, no markdown
- Parse `res.choices[0].message.content`
- **Defensive parsing:** Strip markdown code fences (```` ```json ... ``` ````) before `JSON.parse` — Llama models sometimes wrap output. On parse failure, return a safe default: `{ intent: 'unknown', entities: [], hasFact: false }`

### Step 2.2: Memory Write Agent (`src/agents/memoryWrite.js`)

**Purpose:** After each turn, extract 2–5 atomic facts and store them in the vector store.

**Function signature:** `runMemoryWrite(userMsg, assistantMsg, client, store)` → `string[]` (array of stored facts)

**Implementation:**
- Model: `meta/llama-3.1-8b-instruct`
- System prompt asks for a JSON array of fact strings — provides examples of good vs bad facts
- Input: both the user message and the assistant's response (so it can extract facts from both sides)
- Loop over the parsed array, call `store.storeFact(fact, { source: 'conversation' })` for each
- Same defensive JSON parsing as intake (strip fences, catch errors → return `[]`)

**Critical design decision:** This agent runs **asynchronously** in the pipeline — it must not block the response to the user. It's called with `.catch(console.error)` so failures are logged but don't crash the server.

### Step 2.3: Verification

1. Start a quick test script that creates the OpenAI client, calls `runIntake("My name is Debesh and I use PostgreSQL", client)`
2. Confirm it returns `{ intent: "statement", entities: ["Debesh", "PostgreSQL"], hasFact: true }`
3. Call `runMemoryWrite("My name is Debesh", "Nice to meet you, Debesh!", client, store)`
4. Then `store.retrieveFacts("anything about the user")` — should return facts about Debesh

---

## Phase 3 — Memory Read + Context Packer + Pipeline (Hr 2–3)

### Step 3.1: Memory Read Agent (`src/agents/memoryRead.js`)

**Purpose:** Given a user message, retrieve relevant memories from the vector store.

**Function signature:** `runMemoryRead(message, store, topK=5)` → array of memory objects

**Implementation:**
- Call `store.retrieveFacts(message, topK)`
- Filter results to only those with `score > 0.6` (relevance threshold)
- No LLM call — this is pure vector similarity

### Step 3.2: Context Packer (`src/agents/contextPacker.js`)

**Purpose:** Assemble the final system prompt from memories + recent conversation history.

**Function signature:** `packContext(memories, recentTurns, currentMessage)` → `{ system, messages, memoriesUsed, tokenEstimate }`

**Implementation (pure logic, no LLM):**
- Build a `memoryBlock` string: `## What you remember about this user:\n- fact1\n- fact2...`
- Build a `historyBlock` string: `## Recent conversation:\nuser: ...\nassistant: ...`
- Combine into a system prompt that instructs the LLM to use remembered facts, never contradict them, and give personalised responses
- Return the system prompt, messages array (just `[{ role: 'user', content: currentMessage }]`), the memories used, and a rough token estimate (`systemPrompt.length / 4`)

### Step 3.3: Pipeline (`src/pipeline.js`)

**Purpose:** Wire all 5 agents into a single function that processes one user turn.

**Function signature:** `runPipeline(userMessage, sessionHistory, store, client)` → result object

**Pipeline steps (sequential except Memory Write):**

```
1. Intake Agent        →  intake = runIntake(userMessage, client)
2. Memory Read Agent   →  memories = runMemoryRead(userMessage, store)
3. Context Packer      →  packed = packContext(memories, sessionHistory.slice(-6), userMessage)
4. Main LLM Call       →  model: nvidia/llama-3.1-nemotron-70b-instruct
                          messages: [{ role: 'system', content: packed.system }, ...packed.messages]
                          → rawResponse = res.choices[0].message.content
5. Grounding Agent     →  grounded = runGrounding(rawResponse, memories, client)
6. Memory Write (async) → runMemoryWrite(userMessage, grounded.response, client, store).catch(...)
```

**Return shape:**
```js
{
  response: string,           // the final (possibly corrected) response
  groundingPassed: boolean,   // did grounding check pass?
  contradictions: string[],   // list of contradictions found (empty if passed)
  memoriesUsed: object[],     // which memories were retrieved for this turn
  tokenEstimate: number,      // rough token count of the packed context
  intake: object,             // intake agent output (intent, entities, hasFact)
}
```

### Step 3.4: Verification

Run a terminal chat loop:
1. Send: "My name is Debesh and my team uses PostgreSQL"
2. Send 5+ filler messages
3. Send: "What database do we use?"
4. Confirm the response mentions PostgreSQL (recalled from memory, not conversation window)

---

## Phase 4 — Grounding Agent (Hr 3–4)

### Step 4.1: Grounding Agent (`src/agents/grounding.js`)

**Purpose:** Catch hallucinations and contradictions before the response reaches the user.

**Function signature:** `runGrounding(response, memories, client)` → `{ passed, contradictions, response }`

**Implementation:**
- **Short-circuit:** If `memories.length === 0`, return `{ passed: true, response }` — nothing to ground against
- Model: `nvidia/llama-3.1-nemotron-70b-instruct` (needs strong reasoning for fact-checking)
- System prompt: "You are a fact-checker. Compare an assistant response against known facts. Return JSON only."
- Expected JSON shape: `{ passed: bool, contradictions: string[], correctedResponse: string|null }`
- If `passed === false`, use `correctedResponse` as the final response; otherwise use the original
- Defensive parsing: strip markdown fences, catch errors → default to `{ passed: true, response }`

### Step 4.2: Integrate into pipeline

The grounding step is already accounted for in the pipeline design (Step 3.3, step 5). Ensure:
- Grounding runs **before** Memory Write — so corrected facts get stored, not hallucinated ones
- The pipeline return object includes `groundingPassed` and `contradictions`

### Step 4.3: Verification

1. Manually inject a wrong fact: `store.storeFact("User uses MongoDB for their main database")`
2. The vector store already has "User uses PostgreSQL"
3. Ask: "What database do we use?"
4. The main LLM might say MongoDB (poisoned by the injected fact)
5. Grounding agent should catch the contradiction between the two stored facts, or between the response and the majority of evidence
6. Confirm `groundingPassed === false` and `contradictions` is non-empty

---

## Phase 5 — Express Server + UI (Hr 4–5)

### Step 5.1: Express Server (`src/server.js`)

**Routes:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Main endpoint. Body: `{ message, sessionId?, useContextCore? }`. Runs pipeline or baseline. Returns result + `turnCount`. |
| `GET` | `/memories` | Returns all stored facts as JSON array (for memory inspector). |
| `DELETE` | `/memories` | Deletes the `./vector-store` directory for a clean demo reset. |

**Implementation details:**
- `dotenv/config` imported at top for `.env` loading
- Static files served from `../ui`
- NVIDIA OpenAI client created once at startup
- `sessions` object keyed by `sessionId` — each value is an array of `{ role, content }` turns
- **Baseline mode** (`useContextCore: false`): calls `nvidia/llama-3.1-nemotron-70b-instruct` directly with last 5 turns, no memory/grounding
- After each turn, push both user and assistant messages to `sessions[sessionId]`
- Return `turnCount` in the response for the UI footer

### Step 5.2: Chat UI (`ui/index.html`)

Single HTML file with inline CSS + JS (no build step). Four key sections:

**Layout:**
```
┌──────────────────────────────────────────────────────────────┐
│  [ContextCore ON/OFF toggle]              [Reset Memory btn] │
├──────────────────────────────┬───────────────────────────────┤
│                              │     Memory Inspector          │
│     Chat Window              │                               │
│     (messages scroll here)   │   - stored fact 1             │
│                              │   - stored fact 2 ★ (used)    │
│                              │   - stored fact 3             │
│                              │                               │
├──────────────────────────────┤                               │
│  [input box]     [Send btn]  │                               │
├──────────────────────────────┴───────────────────────────────┤
│  Turn: 12  │  Memories: 8  │  Grounding: ✅ passed           │
└──────────────────────────────────────────────────────────────┘
```

**Features to implement:**
1. **Chat panel** — message bubbles with user/assistant styling, auto-scroll to bottom
2. **ContextCore toggle** — switch at the top, sends `useContextCore: true/false` with each request
3. **Memory Inspector sidebar** — polls `GET /memories` after each turn, highlights which memories were used (compare against `memoriesUsed` in the response)
4. **Status footer** — shows turn count, total memories stored, and grounding status for last turn
5. **Reset button** — calls `DELETE /memories`, clears chat, clears session
6. **Loading state** — show agent activity indicators while waiting: "Retrieving memories...", "Checking for contradictions..."
7. **Contradiction alert** — if `groundingPassed === false`, show the contradictions in a warning banner above the response

**Styling guidance:**
- Dark theme, clean and minimal
- No framework, no build step — vanilla HTML/CSS/JS
- Use `fetch()` for API calls
- Generate a unique `sessionId` per browser tab (use `crypto.randomUUID()`)

### Step 5.3: Verification

1. `npm start` — server should start on port 3000
2. Open `http://localhost:3000` in browser
3. Send a few messages, confirm they appear in the chat
4. Check memory inspector populates after each turn
5. Toggle ContextCore OFF, send the same question — confirm no memories are used
6. Hit Reset, confirm memories clear

---

## Phase 6 — W&B Integration (Hr 5–6)

### Step 6.1: Python eval prerequisites

Create `eval/requirements.txt`:
```
wandb
weave
requests
```

Install: `pip install -r eval/requirements.txt`

### Step 6.2: Option 2 — Memory Quality Evaluation (`eval/eval.py`)

**This is the primary demo visual.** Runs the demo conversation script, then scores both modes on recall questions.

**Script flow:**
1. Define `SETUP_MESSAGES` — the 5 fact-laden turns from the demo script
2. Define `FILLER_MESSAGES` — 10 generic turns to push facts out of the context window
3. Define `EVAL_QUESTIONS` — 4 recall questions with expected keywords:
   - "What database does the user use?" → "postgresql"
   - "What programming language does the team use?" → "typescript"
   - "What did the user fix last week?" → "race condition"
   - "What kind of company is their biggest customer?" → "fintech"
4. For each mode (`contextcore`, `baseline`):
   - `wandb.init(project="contextcore-demo", name=f"eval-{mode}")`
   - Send all setup messages via `POST /chat` with `useContextCore=True` (to populate memories for contextcore mode)
   - Send all filler messages
   - For each eval question: send it, check if `expected` keyword is in the response (case-insensitive)
   - Log each result + final accuracy to W&B
   - `wandb.finish()`
5. Print comparison: `Baseline: 0%  |  ContextCore: 100%`

**Important:** The baseline run should use a **fresh session** (different `sessionId`) and `useContextCore: false`, so it has no access to memories and the setup facts have scrolled out of its 5-turn window.

### Step 6.3: Option 3 — Retrieval Tuning (`eval/retrieval_tune.py`)

**Run this first** (during Phase 1 if time allows) to find optimal retrieval parameters.

**Script flow:**
1. Seed 10 synthetic facts via `POST /chat` (with contextcore ON)
2. Define `RETRIEVAL_TEST_CASES` — queries + expected keywords
3. Sweep over `topK ∈ [3, 5, 8]`, `score_threshold ∈ [0.5, 0.65, 0.75, 0.85]`
4. For each combo, query the pipeline and measure recall (did the expected fact appear?)
5. Log to W&B: `{ topK, score_threshold, recall }`
6. Output the best parameter combo

**Note:** This requires modifying `memoryRead.js` to accept configurable threshold (currently hardcoded to 0.6). Add an optional parameter or expose via a config endpoint.

### Step 6.4: Option 1 — Trace Logging (Optional, if time permits)

Add W&B Weave tracing to the Node.js pipeline. This requires either:
- A Python wrapper that calls the Express API and traces each step, OR
- Adding timing instrumentation directly in `pipeline.js` and logging via a `POST /log` endpoint that the Python script consumes

Simpler approach: add timing to `pipeline.js` directly and include latency data in the `/chat` response. The Python eval script can then log these to W&B.

**Add to pipeline.js return object:**
```js
{
  ...existing fields,
  timing: {
    intake_ms: ...,
    memoryRead_ms: ...,
    llm_ms: ...,
    grounding_ms: ...,
    total_ms: ...,
  }
}
```

---

## Phase 7 — Demo Script + A/B Comparison Mode (Hr 6–7)

### Step 7.1: A/B comparison view in UI

Add a **comparison mode** toggle to the UI. When enabled:
- Each user message is sent **twice**: once with `useContextCore: true`, once with `useContextCore: false` (different session IDs)
- Responses displayed side-by-side in the chat:
  ```
  ┌─────────────────────┬─────────────────────┐
  │  ContextCore ✅      │  Baseline ❌          │
  │  "PostgreSQL, with  │  "I don't have that │
  │   Redis for caching" │   information"       │
  └─────────────────────┴─────────────────────┘
  ```
- Color-code: green border for ContextCore, grey for baseline

### Step 7.2: Demo conversation script

Hard-code the canonical demo script as a "Run Demo" button in the UI:

**Setup turns (1–5):**
```
1. "My name is Aryan. I'm building a B2B SaaS product."
2. "We use PostgreSQL for our main database and Redis for caching."
3. "My team only writes TypeScript. We never use Python."
4. "Last week I fixed a nasty race condition in our auth service."
5. "Our biggest customer is a fintech company with 50k users."
```

**Filler turns (6–15):** 10 generic unrelated messages (weather, jokes, random questions)

**Recall turns (16–25):**
```
16. "What database are we using?"
    → ContextCore: "PostgreSQL, with Redis for caching" ✅
    → Baseline: "I don't have that information" ❌

20. "What language does my team use?"
    → ContextCore: "TypeScript exclusively" ✅
    → Baseline: hallucinates or admits ignorance ❌

25. "Actually we switched to MongoDB last month."
    → Grounding Agent: flags contradiction with stored fact ✅
```

### Step 7.3: "Run Demo" button

Add a button that auto-plays the demo script with ~500ms delay between turns, showing the A/B comparison for recall turns. This is the "wow moment" for the presentation.

---

## Phase 8 — Polish + End-to-End Testing (Hr 7–8)

### Step 8.1: Loading states and agent indicators

In the UI, show real-time pipeline status:
- "Analyzing your message..." (Intake)
- "Retrieving memories..." (Memory Read)
- "Generating response..." (LLM call)
- "Checking for contradictions..." (Grounding)

This requires the server to return which pipeline stages completed. Already handled by the `timing` object added in Phase 6.

### Step 8.2: Error handling

- Graceful fallback if NVIDIA API rate-limits (show user-friendly error)
- Handle empty vector store (first message, no memories yet)
- Handle JSON parse failures from Llama models (already handled with defensive parsing)

### Step 8.3: End-to-end test run

1. Start fresh: `DELETE /memories`, restart server
2. Run the full 25-turn demo script manually
3. Verify:
   - [ ] Turn 16 recall: PostgreSQL + Redis ✅
   - [ ] Turn 20 recall: TypeScript ✅
   - [ ] Turn 25 contradiction detection ✅
   - [ ] Memory inspector shows all stored facts
   - [ ] A/B comparison clearly shows the difference
   - [ ] Pipeline latency < 3 seconds per turn
4. Run W&B eval: `python eval/eval.py`
5. Check W&B dashboard shows the accuracy comparison chart

### Step 8.4: Optional polish (if time permits)

- Record a 2-minute screen capture of the demo
- Export W&B accuracy chart as an image, embed in the UI
- Add a "facts count" badge to the memory inspector header

---

## File Creation Order (Dependency-Aware)

```
1. package.json, .env.example, .gitignore
2. src/store/vectorStore.js          ← no dependencies
3. src/agents/intake.js              ← depends on: OpenAI client (passed in)
4. src/agents/memoryWrite.js         ← depends on: OpenAI client, store
5. src/agents/memoryRead.js          ← depends on: store
6. src/agents/contextPacker.js       ← no dependencies (pure logic)
7. src/agents/grounding.js           ← depends on: OpenAI client
8. src/pipeline.js                   ← depends on: all agents
9. src/server.js                     ← depends on: pipeline, store
10. ui/index.html                    ← depends on: server API
11. eval/requirements.txt            ← no dependencies
12. eval/eval.py                     ← depends on: running server
13. eval/retrieval_tune.py           ← depends on: running server
```

---

## Models Used (NVIDIA NIM Free Tier)

| Agent | Model | Why |
|-------|-------|-----|
| Intake | `meta/llama-3.1-8b-instruct` | Fast, cheap. Simple JSON extraction. |
| Memory Write | `meta/llama-3.1-8b-instruct` | Fast, cheap. Fact extraction from text. |
| Main LLM | `nvidia/llama-3.1-nemotron-70b-instruct` | Strong reasoning for natural conversation. |
| Grounding | `nvidia/llama-3.1-nemotron-70b-instruct` | Needs strong reasoning to detect contradictions. |
| Embeddings | `nvidia/nv-embedqa-e5-v5` | Purpose-built for retrieval/QA embeddings. |

All accessed via OpenAI-compatible API at `https://integrate.api.nvidia.com/v1`.

---

## Definition of Done (from PRD)

- [ ] 50-turn conversation holds full context — nothing from turn 1–5 is forgotten by turn 40
- [ ] Grounding agent catches 3/3 injected contradictions
- [ ] A/B comparison is visually clear in the UI
- [ ] W&B eval run shows ContextCore vs baseline accuracy side-by-side
- [ ] Pipeline latency is under 3 seconds end-to-end per turn
- [ ] Memory inspector shows which facts were retrieved for each response

---

## Verification Strategy

1. **Unit-level:** After each phase, run the verification step described in that phase
2. **Integration:** After Phase 5, run the full 25-turn demo script manually in the browser
3. **Evaluation:** After Phase 6, run `python eval/eval.py` and confirm W&B dashboard shows results
4. **End-to-end:** Phase 8 full run-through with all features enabled
