# ContextCore — Multi-Agent Context Manager

## What Problem It Solves

Every LLM API call is stateless. When you maintain a conversation, you manually pass the message history back on every request. This works fine for a few turns, but has two compounding problems:

1. **Context window overflow.** The model has a hard token limit (32,768 for Qwen3-14B). Once your history exceeds it, you must drop messages. The standard fix — a sliding window — means old facts are silently forgotten.
2. **Noise.** Passing the full raw history sends everything to the model regardless of relevance. A question about your database stack doesn't need 40 turns of unrelated conversation in the prompt.

ContextCore replaces the sliding window with a 7-agent pipeline. Instead of passing raw history, it extracts atomic facts into a vector store after each turn, and retrieves only the relevant ones at query time. The model always sees **the full conversation history** plus **the most relevant facts from any prior turn** — no information is ever dropped.

---

## Pipeline Overview

```
User Message
     │
     ▼
┌─────────────────────────┐
│  1. Understanding        │  Classifies intent, expands retrieval query
└──────────┬──────────────┘
           │ { retrievalQuery, needsContext, intent }
           ▼
┌─────────────────────────┐
│  2. Memory Read          │  Vector similarity search → top-K candidates
└──────────┬──────────────┘
           │ candidates (score > 0.2) — or skip if needsContext=false
           ▼
┌─────────────────────────┐
│  3. Relevance Decision   │  LLM filters candidates to only what's needed
└──────────┬──────────────┘
           │ filteredMemories
           ▼
┌─────────────────────────┐
│  4. Context Packer       │  Assembles system prompt: facts + full history
└──────────┬──────────────┘
           │ { system, messages, tokenEstimate }
           ▼
┌─────────────────────────┐
│  5. Main LLM Call        │  Qwen3-14B-Instruct generates the response
└──────────┬──────────────┘
           │ rawResponse
           ▼
┌─────────────────────────┐
│  6. Grounding Agent      │  Fact-checks response against retrieved memories
└──────────┬──────────────┘
           │ { passed, contradictions, correctedResponse }
           ▼
┌─────────────────────────┐   (async — never blocks the response)
│  7. Memory Write         │  Extracts 2–5 atomic facts, stores in vector DB
└─────────────────────────┘
           │
           ▼
     Response to User
```

**Total LLM calls per turn:** 4 (Understanding, Relevance Decision, Main LLM, Grounding) + 1 async (Memory Write) = 5

---

## Agent-by-Agent Breakdown

### Agent 1 — Understanding (`src/agents/understanding.js`)

**Purpose:** Decide whether memory retrieval is needed, and if so, expand the user's query into a better search string.

**Why it exists:** A user might say "what was that thing I mentioned?" — a terrible embedding query. Or they might ask "what is TCP/IP?" — a self-contained question that has nothing to do with their stored facts. Blindly searching memory on every message wastes LLM calls and injects noise.

**How it works:**
- Receives the user's latest message and the last 4 turns of conversation (for context on pronouns/references)
- Calls the LLM with a strict JSON schema prompt
- Returns three fields:
  - `retrievalQuery` — an expanded version of the message optimized for semantic search (e.g., "what database do we use?" → "user's database stack technology choice")
  - `needsContext` — `false` only for fully self-contained questions (math, trivia, general knowledge); defaults to `true` on any ambiguity
  - `intent` — `question | statement | command | chitchat`

**Fail-safe:** On JSON parse error, returns `{ retrievalQuery: originalMessage, needsContext: true }` — retrieval still runs.

```
Input:  "what database are we using again?"
Output: {
  retrievalQuery: "user's database technology stack storage choice",
  needsContext: true,
  intent: "question"
}
```

---

### Agent 2 — Memory Read (`src/agents/memoryRead.js`)

**Purpose:** Retrieve the top-K most semantically similar facts from the vector store.

**How it works:**
- Takes the `retrievalQuery` from Agent 1
- Queries the vector store with cosine similarity
- Filters out candidates below score `0.2` (threshold is low because local hash embeddings score lower than neural embeddings)
- Returns up to 5 candidates

**Skipped entirely** when `needsContext=false` from Agent 1.

**No LLM call.** This is a pure vector similarity lookup — fast and cheap.

---

### Agent 3 — Relevance Decision (`src/agents/relevanceDecision.js`)

**Purpose:** A second filter — use the LLM to read the candidates and decide which ones actually answer the question.

**Why it exists:** Vector similarity finds lexically related facts, but "related" doesn't mean "needed." If the user asks "what database do I use?" and memory contains both "User uses PostgreSQL" and "User fixed a race condition in auth service", both might score well because "service" and "database" share context — but only one is actually relevant.

**How it works:**
- Formats all candidates as a numbered list
- Asks the LLM: given this question and these facts, which ones (if any) are actually needed to answer it?
- Returns `{ useContext, filteredMemories }` — only the indices the LLM selects pass through

**Fail-safe:** On JSON parse error, passes through all candidates unchanged (conservative — better to over-inject than to silently drop relevant facts).

```
Candidates passed in:
  [0] User uses PostgreSQL and Redis
  [1] User fixed a race condition in auth service
  [2] User's team writes TypeScript

Question: "what database are we using?"

Output:
  { useContext: true, relevantIndices: [0] }
  → only "User uses PostgreSQL and Redis" passes through
```

---

### Agent 4 — Context Packer (`src/agents/contextPacker.js`)

**Purpose:** Assemble the final prompt that will be sent to the main LLM.

**How it works (pure logic, no LLM call):**
1. Takes the filtered memories from Agent 3 and the full session history
2. Builds a system prompt with two sections:
   - `## Relevant background` — the retrieved facts, with an instruction to use them only if directly relevant
   - `## Recent conversation` — the full conversation history for conversational continuity
3. The user's current message becomes the sole user turn in the messages array

**The output is deliberately minimal:** only one `user` message in the `messages` array, everything else in the system prompt. This ensures the model focuses on the current question, not re-processing the whole conversation.

```
System prompt structure:
  You are a helpful assistant. [behavior rules]
  Use the background facts below only if the question directly requires them.

  ## Relevant background:
  - User uses PostgreSQL and Redis

  ## Recent conversation:
  user: My team only writes TypeScript.
  assistant: Understood. TypeScript is a great choice for type safety...
  user: We fixed a race condition in auth last week.
  assistant: ...

Messages:
  [{ role: "user", content: "what database are we using again?" }]
```

---

### Agent 5 — Main LLM Call (`src/pipeline.js`)

**Purpose:** Generate the actual response.

**Model:** `OpenPipe/Qwen3-14B-Instruct` via WandB Serverless Inference (`https://api.inference.wandb.ai/v1`)

**Input:** The packed context from Agent 4 — system prompt with retrieved memories and full history, plus the current user message.

**max_tokens:** 1000 — enough for technical explanations, short enough to not blow the context budget.

No special prompting tricks here. The quality of the response depends on the upstream agents delivering exactly the right context.

---

### Agent 6 — Grounding (`src/agents/grounding.js`)

**Purpose:** Catch hallucinations and factual contradictions before the response reaches the user.

**How it works:**
- Compares the raw LLM response against the retrieved facts from Agent 3
- Returns `{ passed, contradictions[], correctedResponse }`
- If `passed=false`, the corrected response replaces the original
- The UI shows a red warning banner listing the specific contradictions

**Skipped entirely** when no memories were retrieved (nothing to ground against).

**Example trigger:**
```
Stored fact:  "User's team writes TypeScript"
LLM response: "...your team uses Python for backend services..."
Grounding:    passed=false
              contradictions: ["Response claims Python is used; stored fact says TypeScript"]
              correctedResponse: "...your team uses TypeScript..."
```

---

### Agent 7 — Memory Write (`src/agents/memoryWrite.js`)

**Purpose:** Extract durable facts from the completed turn and persist them to the vector store for future retrieval.

**Runs async** — it does not block the response. The user gets their answer immediately; memory is updated in the background.

**How it works:**
- Receives the user's message and the grounded response
- Asks the LLM to extract 2–5 atomic, self-contained facts worth remembering
- Each fact is independently embedded and stored in the vector store with a timestamp
- Facts must be specific and reusable (e.g., "User is building a B2B SaaS product"), not pleasantries or transient remarks

**Atomic storage matters:** Each fact is stored separately, not as a raw message blob. This means future queries can retrieve a specific fact without pulling in unrelated facts from the same turn.

```
Turn:
  User:      "My team only writes TypeScript. We never use Python."
  Assistant: "Got it — TypeScript only. I'll keep that in mind."

Extracted facts stored:
  → "User's team uses TypeScript exclusively"
  → "User's team does not use Python"
```

---

## The Vector Store (`src/store/vectorStore.js`)

The vector store is file-backed (via `vectra`, stored in `./vector-store/`). It persists across server restarts and survives conversation resets unless explicitly cleared via `DELETE /memories`.

### Embedding: Local Feature Hashing

WandB Serverless Inference has no embedding endpoint. Instead of an API call, ContextCore uses a local hash-based embedding:

1. Text is lowercased, punctuation stripped, stop words removed
2. Each remaining word is hashed with DJB2 into a 512-dimensional bucket: `vec[hash(word) % 512] += 1`
3. Adjacent word pairs (bigrams) are also hashed: `vec[hash("word1_word2") % 512] += 0.5` — this captures short phrases
4. The vector is L2-normalized to unit length
5. Similarity is cosine distance between the normalized vectors

**Trade-off:** This is far weaker than a neural embedding model (no synonyms, no semantic distance). "PostgreSQL" and "database" won't match unless both words appear together. This is why the score threshold is set low at `0.2` and why Agent 1 expands the query — the expansion injects synonyms that the hash embedding can then match on.

### Operations

| Function | Description |
|---|---|
| `storeFact(text, metadata)` | Embeds text, writes to disk with timestamp |
| `retrieveFacts(query, topK)` | Embeds query, returns top-K by cosine similarity |
| `getAllFacts()` | Returns all stored facts (for the Memory Inspector UI) |

Lazy initialization via `indexReadyPromise` prevents race conditions if multiple facts are written simultaneously.

---

## Comparison: ContextCore vs Baseline

| | Baseline | ContextCore |
|---|---|---|
| History passed to LLM | Last 6 messages (3 turns) | Full session history |
| Fact recall beyond 3 turns | No — forgotten | Yes — retrieved from vector store |
| Context window overflow | Silently truncates | Never truncates (full history + injected facts) |
| Hallucination detection | None | Grounding agent corrects before delivery |
| Per-turn LLM calls | 1 | 5 (4 sync + 1 async) |
| Latency overhead | ~0ms | ~3–8s on WandB free tier |

---

## Data Flow for a Concrete Example

**Turn 1 (setup):** "My name is Aryan. I'm building a B2B SaaS product."

1. Understanding → `needsContext=false` (statement, nothing to retrieve yet)
2. Memory Read → skipped
3. Relevance Decision → skipped
4. Context Packer → system prompt with empty memory block, empty history
5. Main LLM → "Got it, Aryan! Happy to help with your B2B SaaS project."
6. Grounding → skipped (no memories to check against)
7. Memory Write (async) → stores: `["User's name is Aryan", "User is building a B2B SaaS product"]`

---

**Turn 8 (after 6 unrelated filler turns):** "What project am I working on?"

1. Understanding → `retrievalQuery: "user's project name type of product"`, `needsContext: true`
2. Memory Read → retrieves top-5 facts; "User is building a B2B SaaS product" scores ~0.45
3. Relevance Decision → selects index 0 only ("User is building a B2B SaaS product")
4. Context Packer → injects that fact into system prompt + full 14-message history
5. Main LLM → "You're working on a B2B SaaS product."
6. Grounding → passes (response matches stored fact)
7. Memory Write (async) → may store additional facts from this exchange

**The baseline** (3-turn window) has already dropped Turn 1. It would answer: "I don't have information about your specific project from our current conversation."

---

## File Map

```
src/
├── pipeline.js              Orchestrates all 7 steps, collects timing per step
├── server.js                Express HTTP server — /chat, /memories, /retrieve
├── agents/
│   ├── understanding.js     Agent 1 — query expansion + intent classification
│   ├── memoryRead.js        Agent 2 — vector similarity retrieval
│   ├── relevanceDecision.js Agent 3 — LLM-based relevance filter
│   ├── contextPacker.js     Agent 4 — prompt assembly (no LLM)
│   ├── grounding.js         Agent 6 — hallucination / contradiction detection
│   └── memoryWrite.js       Agent 7 — atomic fact extraction + persistence
└── store/
    └── vectorStore.js       Local hash embeddings + vectra file-backed index
```
