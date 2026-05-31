# ContextCore — Builder PRD
**8-Hour Prototype · Internal**

---

## What We're Building

A multi-agent middleware layer that intercepts LLM chat messages, manages memory via a vector store, and prevents hallucination through a grounding agent. The user chats through a simple UI; ContextCore handles everything behind it invisibly.

**End state:** A live demo where a user can have a 50-turn conversation, and ContextCore correctly recalls facts from turn 2 at turn 48 — while the baseline (no middleware) fails.

---

## System Architecture

```
User Message
     │
     ▼
┌─────────────────────────────────────────────────┐
│              ContextCore Middleware              │
│                                                  │
│  [1. Intake Agent]                               │
│       │ parse intent + extract entities          │
│       ▼                                          │
│  [2. Memory Read Agent]                          │
│       │ semantic search → top-K memories         │
│       ▼                                          │
│  [3. Context Packer]                             │
│       │ system prompt + memories + last N turns  │
│       ▼                                          │
│  [LLM Call → Claude]                             │
│       │ raw response                             │
│       ▼                                          │
│  [4. Grounding Agent]                            │
│       │ validate against stored facts            │
│       ▼                                          │
│  [5. Memory Write Agent]  ←── runs async         │
│       │ extract + store new facts                │
└─────────────────────────────────────────────────┘
     │
     ▼
 Response to User
```

**5 agents. 2 use Llama 8B via NVIDIA NIM (Intake + Memory Write). 1 is pure cosine similarity (Memory Read). 1 is logic only (Context Packer). 1 uses Nemotron 70B (Grounding).**

---

## Hour-by-Hour Build Plan

### Hr 0–1 · Repo + Vector Store

**Goal:** Can embed a string and retrieve it by semantic similarity.

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
└── package.json
```

**Install:**
```bash
npm init -y
npm install openai express cors vectra dotenv
```

**`vectorStore.js` — in-memory only, no external DB:**
```js
import { LocalIndex } from 'vectra';
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: process.env.NVIDIA_API_KEY,
  baseURL: 'https://integrate.api.nvidia.com/v1',
});
const index = new LocalIndex('./vector-store');

export async function embedText(text) {
  const res = await client.embeddings.create({
    model: 'nvidia/nv-embedqa-e5-v5',
    input: text,
    encoding_format: 'float',
  });
  return res.data[0].embedding;
}

export async function storeFact(fact, metadata = {}) {
  const vector = await embedText(fact);
  await index.insertItem({ vector, metadata: { text: fact, ts: Date.now(), ...metadata } });
}

export async function retrieveFacts(query, topK = 5) {
  const vector = await embedText(query);
  const results = await index.queryItems(vector, topK);
  return results.map(r => ({ text: r.item.metadata.text, score: r.score, ...r.item.metadata }));
}
```

> **Done when:** `storeFact("user prefers TypeScript")` → `retrieveFacts("what language does the user like?")` returns it with score > 0.8

---

### Hr 1–2 · Intake + Memory Write Agent

**Goal:** A single turn produces stored facts in the vector store.

**`intake.js`** — classify + extract entities from incoming message:
```js
export async function runIntake(message, client) {
  const res = await client.chat.completions.create({
    model: 'meta/llama-3.1-8b-instruct',
    max_tokens: 200,
    messages: [
      { role: 'system', content: `Extract structured info from this chat message. Return JSON only:
{
  "intent": "question|statement|command|chitchat",
  "entities": ["list of named entities, facts, preferences mentioned"],
  "hasFact": true/false
}` },
      { role: 'user', content: message }
    ]
  });
  return JSON.parse(res.choices[0].message.content);
}
```

**`memoryWrite.js`** — extract atomic facts and store them:
```js
export async function runMemoryWrite(userMsg, assistantMsg, client, store) {
  const res = await client.chat.completions.create({
    model: 'meta/llama-3.1-8b-instruct',
    max_tokens: 400,
    messages: [
      { role: 'system', content: `Given a conversation turn, extract 2–5 atomic facts worth remembering long-term.
Facts should be: specific, self-contained, and useful for future context.
Return JSON array only: ["fact 1", "fact 2", ...]
Skip pleasantries, greetings, and non-informational content.

Examples of good facts:
- "User's name is Aryan"
- "User is building a B2B SaaS product"
- "User's team uses TypeScript and PostgreSQL"
- "User fixed a race condition bug in their auth service last week"` },
      { role: 'user', content: `User said: "${userMsg}"\nAssistant said: "${assistantMsg}"` }
    ]
  });

  const facts = JSON.parse(res.choices[0].message.content);
  for (const fact of facts) {
    await store.storeFact(fact, { source: 'conversation' });
  }
  return facts;
}
```

> **Done when:** After one message about yourself, `retrieveFacts("anything about the user")` returns 2–5 relevant facts.

---

### Hr 2–3 · Memory Read + Context Packer

**Goal:** Full round-trip. User message in → context-packed LLM response out.

**`memoryRead.js`:**
```js
export async function runMemoryRead(message, store, topK = 5) {
  const memories = await store.retrieveFacts(message, topK);
  // Also pull recent high-score memories regardless of query (recency bonus)
  return memories.filter(m => m.score > 0.6);
}
```

**`contextPacker.js`** — assembles the final prompt. No LLM call, pure logic:
```js
export function packContext(memories, recentTurns, currentMessage) {
  const memoryBlock = memories.length > 0
    ? `## What you remember about this user:\n${memories.map(m => `- ${m.text}`).join('\n')}`
    : '';

  const historyBlock = recentTurns.length > 0
    ? `## Recent conversation:\n${recentTurns.map(t => `${t.role}: ${t.content}`).join('\n')}`
    : '';

  const systemPrompt = `You are a helpful assistant with long-term memory.
Use the remembered facts below to give accurate, personalised responses.
Never contradict the remembered facts unless the user explicitly updates them.

${memoryBlock}

${historyBlock}`.trim();

  return {
    system: systemPrompt,
    messages: [{ role: 'user', content: currentMessage }],
    memoriesUsed: memories,
    tokenEstimate: systemPrompt.length / 4, // rough estimate
  };
}
```

**`pipeline.js`** — wire it all together:
```js
export async function runPipeline(userMessage, sessionHistory, store, client) {
  // 1. Intake
  const intake = await runIntake(userMessage, client);

  // 2. Memory Read
  const memories = await runMemoryRead(userMessage, store);

  // 3. Pack context
  const recentTurns = sessionHistory.slice(-6); // last 3 exchanges
  const packed = packContext(memories, recentTurns, userMessage);

  // 4. LLM call
  const llmRes = await client.chat.completions.create({
    model: 'nvidia/llama-3.1-nemotron-70b-instruct',
    max_tokens: 1000,
    messages: [
      { role: 'system', content: packed.system },
      ...packed.messages,
    ],
  });
  const response = llmRes.choices[0].message.content;

  // 5. Memory Write (async — don't block the response)
  runMemoryWrite(userMessage, response, client, store).catch(console.error);

  return { response, memoriesUsed: packed.memoriesUsed, intake };
}
```

> **Done when:** Chat in terminal. Tell it your name and DB preference. 10 messages later, ask "what DB do we use?" → correct answer.

---

### Hr 3–4 · Grounding Agent

**Goal:** Catch and handle contradictions before they reach the user.

**`grounding.js`:**
```js
export async function runGrounding(response, memories, client) {
  if (memories.length === 0) return { passed: true, response };

  const memoryContext = memories.map(m => `- ${m.text}`).join('\n');

  const res = await client.chat.completions.create({
    model: 'nvidia/llama-3.1-nemotron-70b-instruct',
    max_tokens: 600,
    messages: [
      { role: 'system', content: `You are a fact-checker. Compare an assistant response against known facts.
Return JSON only:
{
  "passed": true/false,
  "contradictions": ["describe any contradiction found"],
  "correctedResponse": "corrected version if needed, or null if passed"
}` },
      { role: 'user', content: `Known facts:\n${memoryContext}\n\nAssistant response to check:\n"${response}"` }
    ]
  });

  const result = JSON.parse(res.choices[0].message.content);
  return {
    passed: result.passed,
    contradictions: result.contradictions || [],
    response: result.correctedResponse || response,
  };
}
```

**Plug into pipeline.js** — replace the return block:
```js
  // 4b. Grounding check
  const grounded = await runGrounding(response, memories, client);

  return {
    response: grounded.response,
    groundingPassed: grounded.passed,
    contradictions: grounded.contradictions,
    memoriesUsed: packed.memoriesUsed,
    intake,
  };
```

> **Done when:** Manually inject a wrong fact ("user uses MongoDB") into vector store, then ask about the DB. Grounding agent catches the contradiction.

---

### Hr 4–5 · Express Server + Basic UI

**`server.js`:**
```js
import express from 'express';
import cors from 'cors';
import OpenAI from 'openai';
import { createStore } from './store/vectorStore.js';
import { runPipeline } from './pipeline.js';

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('../ui'));

const client = new OpenAI({
  apiKey: process.env.NVIDIA_API_KEY,
  baseURL: 'https://integrate.api.nvidia.com/v1',
});
const store = createStore();

// Per-session history (in-memory, keyed by sessionId)
const sessions = {};

app.post('/chat', async (req, res) => {
  const { message, sessionId = 'default', useContextCore = true } = req.body;
  if (!sessions[sessionId]) sessions[sessionId] = [];

  let result;
  if (useContextCore) {
    result = await runPipeline(message, sessions[sessionId], store, client);
  } else {
    // Baseline: raw last-5 turns only, no memory
    const raw = await client.chat.completions.create({
      model: 'nvidia/llama-3.1-nemotron-70b-instruct',
      max_tokens: 1000,
      messages: [...sessions[sessionId].slice(-5), { role: 'user', content: message }],
    });
    result = { response: raw.choices[0].message.content, memoriesUsed: [], groundingPassed: null };
  }

  sessions[sessionId].push({ role: 'user', content: message });
  sessions[sessionId].push({ role: 'assistant', content: result.response });

  res.json(result);
});

app.get('/memories', async (req, res) => {
  const all = await store.getAllFacts();
  res.json(all);
});

app.listen(3000, () => console.log('ContextCore running on :3000'));
```

**`ui/index.html`** — minimal but functional, with memory inspector panel and A/B toggle:

Key UI elements to build:
- Chat window (left, full height)
- Memory Inspector sidebar (right) — live list of stored facts, highlights which were used in last turn
- Toggle at top: `ContextCore ON / OFF`
- Turn counter + token saved counter in footer

> **Done when:** Can have a full conversation in browser, see memories populate in sidebar.

---

### Hr 5–6 · W&B Integration (see options below)

> See **W&B Integration Options** section — pick one and implement.

---

### Hr 6–7 · Demo Script + A/B Mode

Wire up the `useContextCore: false` path fully. Build the comparison view — side by side or sequential answers shown for the same question.

**Demo conversation script** (run this to generate the "wow moment"):

```
Turn 1:  "My name is Aryan. I'm building a B2B SaaS product."
Turn 2:  "We use PostgreSQL for our main database and Redis for caching."
Turn 3:  "My team only writes TypeScript. We never use Python."
Turn 4:  "Last week I fixed a nasty race condition in our auth service."
Turn 5:  "Our biggest customer is a fintech company with 50k users."
...
[10 filler turns about unrelated things]
...
Turn 16: "What database are we using?" 
         → ContextCore: "PostgreSQL, with Redis for caching" ✅
         → Baseline:    "I don't have that information" ❌

Turn 20: "What language does my team use?"
         → ContextCore: "TypeScript exclusively" ✅
         → Baseline:    hallucinates or admits ignorance ❌

Turn 25: "Actually we switched to MongoDB last month."
         → Grounding Agent: flags contradiction with stored fact, asks to confirm update ✅
```

---

### Hr 7–8 · Polish + End-to-End Test

- Add loading spinners + agent activity indicators ("Retrieving memories...", "Checking for contradictions...")
- Run the full demo script once clean
- Fix anything broken
- Optional: record a 2-min screen capture

---

## W&B Integration Options

Three distinct ways to use Weights & Biases. **Pick one for the demo** — Option 2 gives the best visual payoff.

---

### Option 1 · Trace Logging (Observability)
**Best for:** Debugging the pipeline, showing latency per agent.

Log every pipeline run as a W&B trace with timing and token counts per agent.

```bash
pip install wandb weave
```

```python
import weave
import wandb

wandb.init(project="contextcore-demo")

@weave.op()
def run_pipeline_traced(user_message, session_id):
    start = time.time()

    intake_result = run_intake(user_message)          # traced
    memories = run_memory_read(user_message)           # traced
    packed = pack_context(memories, history, user_message)
    llm_response = call_llm(packed)                   # traced
    grounding = run_grounding(llm_response, memories) # traced

    wandb.log({
        "turn": turn_count,
        "memories_retrieved": len(memories),
        "top_memory_score": memories[0]["score"] if memories else 0,
        "grounding_passed": grounding["passed"],
        "tokens_in_context": packed["tokenEstimate"],
        "latency_ms": (time.time() - start) * 1000,
    })

    return grounding["response"]
```

**What you see in W&B:** Timeline of every agent call, latency breakdown, token usage over turns, grounding pass/fail rate.

---

### Option 2 · Memory Quality Evaluation (Recommended)
**Best for:** Quantitatively proving ContextCore works vs baseline.

Run an automated evaluation at the end of the demo conversation. Score both modes on the same set of recall questions.

```python
import wandb

EVAL_QUESTIONS = [
    { "question": "What database does the user use?",           "expected": "postgresql" },
    { "question": "What programming language does the team use?", "expected": "typescript" },
    { "question": "What did the user fix last week?",           "expected": "race condition" },
    { "question": "What kind of company is their biggest customer?", "expected": "fintech" },
]

def run_eval(mode="contextcore"):
    wandb.init(project="contextcore-demo", name=f"eval-{mode}")
    scores = []

    for q in EVAL_QUESTIONS:
        response = query_pipeline(q["question"], use_context_core=(mode == "contextcore"))
        passed = q["expected"].lower() in response.lower()
        scores.append(passed)

        wandb.log({
            "question": q["question"],
            "expected": q["expected"],
            "response": response,
            "passed": passed,
            "mode": mode,
        })

    wandb.log({ "accuracy": sum(scores) / len(scores), "mode": mode })
    wandb.finish()
    return sum(scores) / len(scores)

baseline_score  = run_eval("baseline")      # logs to W&B
contextcore_score = run_eval("contextcore") # logs to W&B

print(f"Baseline:     {baseline_score:.0%}")   # e.g. 0%
print(f"ContextCore:  {contextcore_score:.0%}") # e.g. 100%
```

**What you see in W&B:** A Table showing each recall question with pass/fail per mode. Bar chart comparing accuracy: 0% vs 100%. This is the demo's most compelling visual — export it as an image and put it in the demo UI.

---

### Option 3 · Retrieval Quality Scoring (Deep Eval)
**Best for:** Tuning the vector store — finding the right topK, score threshold, and recency weighting.

```python
import wandb

wandb.init(project="contextcore-retrieval-tuning")

# Sweep over retrieval parameters
for top_k in [3, 5, 8]:
    for score_threshold in [0.5, 0.65, 0.75, 0.85]:
        for recency_weight in [0.0, 0.2, 0.4]:

            hits = 0
            for test in RETRIEVAL_TEST_CASES:
                results = retrieve_facts(test["query"], top_k, score_threshold, recency_weight)
                retrieved_texts = [r["text"] for r in results]
                if any(test["expected_keyword"] in t for t in retrieved_texts):
                    hits += 1

            recall = hits / len(RETRIEVAL_TEST_CASES)

            wandb.log({
                "top_k": top_k,
                "score_threshold": score_threshold,
                "recency_weight": recency_weight,
                "recall": recall,
            })

wandb.finish()
```

**What you see in W&B:** A parameter sweep table. Find the topK + threshold combo that maximises recall. Run this in hour 1 with synthetic data, then hardcode the winning params before the real demo.

---

### Recommendation for the 8-Hour Build

| Hour | W&B Activity |
|------|-------------|
| Hr 0–1 | Run **Option 3** (retrieval tuning) with 10 synthetic facts — 20 min, find best params |
| Hr 5–6 | Implement **Option 2** (eval harness) — 60 min, produces the demo's comparison visual |
| Hr 6–7 | Add **Option 1** trace logging — 30 min, nice to have for the pipeline walkthrough |

All three together take ~2 hours and cover observability, evaluation, and tuning — giving W&B a real role at every layer of the system, not just bolted on.

---

## Stack Summary

| Layer | Tool |
|-------|------|
| LLM | NVIDIA NIM (`meta/llama-3.1-8b-instruct` for agents, `nvidia/llama-3.1-nemotron-70b-instruct` for grounding + main) |
| Vector Store | `vectra` (in-memory, no infra needed) |
| Embeddings | NVIDIA NIM `nvidia/nv-embedqa-e5-v5` |
| Server | Express.js |
| UI | Plain HTML/JS (no build step) |
| Eval + Observability | Weights & Biases (`wandb` + `weave`) |
| API Key | `NVIDIA_API_KEY` from https://build.nvidia.com |
| Runtime | Node.js 20+ / Python 3.11+ (W&B scripts) |

---

## Definition of Done

The prototype is complete when:

- [ ] 50-turn conversation holds full context — nothing from turn 1–5 is forgotten by turn 40
- [ ] Grounding agent catches 3/3 injected contradictions
- [ ] A/B comparison is visually clear in the UI (or terminal)
- [ ] W&B eval run shows ContextCore vs baseline accuracy side-by-side
- [ ] Pipeline latency is under 3 seconds end-to-end per turn
- [ ] Memory inspector shows which facts were retrieved for each response