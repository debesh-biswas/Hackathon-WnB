# WandB Migration — Changes to Re-apply After Force Pull

This file documents every change made to switch the backend from NVIDIA NIM to
WandB Serverless Inference. Re-apply these after a `git pull --force`.

---

## 1. `src/server.js`

**a) Add `File` polyfill import at the very top (first line):**
```js
import 'dotenv/config';
```
No change needed here — the polyfill is handled by `setup.cjs` (see section 6).

**b) Change the OpenAI client credentials:**
```js
// BEFORE
const client = new OpenAI({
  apiKey: process.env.NVIDIA_API_KEY,
  baseURL: 'https://integrate.api.nvidia.com/v1',
});

// AFTER
const client = new OpenAI({
  apiKey: process.env.WANDB_API_KEY,
  baseURL: 'https://api.inference.wandb.ai/v1',
});
```

**c) Change the baseline model name (appears once in the `/chat` route):**
```js
// BEFORE
model: 'google/gemma-3n-e4b-it',

// AFTER
model: 'OpenPipe/Qwen3-14B-Instruct',
```

---

## 2. `src/store/vectorStore.js`

Replace the entire file with the version below.
Key changes: remove OpenAI/NVIDIA embedding client; replace with local
feature-hashing embedding (no external API needed).

```js
import { LocalIndex } from 'vectra';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// WandB inference has no embedding endpoint, so we use local feature-hashing:
// each word/bigram is hashed into a 512-dim bucket vector weighted by TF.
// Cosine similarity on these vectors correlates with keyword overlap, which is
// sufficient for short factual queries in this demo.
const EMBED_DIM = 512;

const STOP_WORDS = new Set([
  'i','me','my','we','our','you','your','he','his','she','her','it','its',
  'they','their','what','which','who','this','that','these','those',
  'am','is','are','was','were','be','been','being','have','has','had',
  'do','does','did','will','would','could','should','may','might','shall',
  'a','an','the','and','but','or','nor','for','so','yet',
  'in','on','at','to','of','with','by','from','up','about','into','than',
  'just','only','also','very','can','not','no','how',
]);

function djb2(str) {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash + str.charCodeAt(i)) & 0x7fffffff;
  }
  return hash;
}

function embedText(text) {
  const vec = new Array(EMBED_DIM).fill(0);
  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 1 && !STOP_WORDS.has(w));

  for (const w of words) {
    vec[djb2(w) % EMBED_DIM] += 1;
  }
  for (let i = 0; i < words.length - 1; i++) {
    vec[djb2(words[i] + '_' + words[i + 1]) % EMBED_DIM] += 0.5;
  }

  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return norm > 0 ? vec.map(v => v / norm) : vec;
}

export function createStore() {
  const index = new LocalIndex(join(__dirname, '../../vector-store'));

  let indexReadyPromise = null;
  async function ensureIndex() {
    if (!indexReadyPromise) {
      indexReadyPromise = (async () => {
        if (!await index.isIndexCreated()) {
          await index.createIndex();
        }
      })();
    }
    return indexReadyPromise;
  }

  async function storeFact(fact, metadata = {}) {
    await ensureIndex();
    const vector = embedText(fact);
    await index.insertItem({
      vector,
      metadata: { text: fact, ts: Date.now(), ...metadata },
    });
  }

  async function retrieveFacts(query, topK = 5) {
    await ensureIndex();
    const vector = embedText(query);
    const results = await index.queryItems(vector, topK);
    return results.map(({ item, score }) => {
      const { text, ts, ...rest } = item.metadata;
      return { text, score, ts, ...rest };
    });
  }

  async function getAllFacts() {
    await ensureIndex();
    const items = await index.listItems();
    return items.map((item) => item.metadata);
  }

  return { storeFact, retrieveFacts, getAllFacts };
}
```

---

## 3. `src/agents/intake.js`

Change model name (line ~43):
```js
// BEFORE
model: 'google/gemma-3n-e4b-it',

// AFTER
model: 'OpenPipe/Qwen3-14B-Instruct',
```

---

## 4. `src/agents/grounding.js`

Change model name (line ~7):
```js
// BEFORE
model: 'google/gemma-3n-e4b-it',

// AFTER
model: 'OpenPipe/Qwen3-14B-Instruct',
```

---

## 5. `src/agents/memoryWrite.js`

Change model name (line ~49):
```js
// BEFORE
model: 'google/gemma-3n-e4b-it',

// AFTER
model: 'OpenPipe/Qwen3-14B-Instruct',
```

---

## 6. `src/pipeline.js`

Change model name (line ~27):
```js
// BEFORE
model: 'google/gemma-3n-e4b-it',

// AFTER
model: 'OpenPipe/Qwen3-14B-Instruct',
```

---

## 7. `src/agents/memoryRead.js`

Lower the score threshold — hash embeddings score lower than neural embeddings:
```js
// BEFORE
return candidates.filter(m => m.score > 0.6);

// AFTER
return candidates.filter(m => m.score > 0.2);
```

---

## 8. New file: `setup.cjs` (create at repo root)

Node 18.15 has `File` in `node:buffer` but doesn't expose it as a global until
18.17. The `undici` dependency (used by the openai SDK and cheerio) references
`globalThis.File` at import time, causing a crash. This CJS preload runs
synchronously before any ESM module is parsed.

```js
// setup.cjs
const { File } = require('buffer');
if (typeof globalThis.File === 'undefined') globalThis.File = File;
```

---

## 9. New file: `eval/test_context_overflow.py` (create in eval/)

See the full file in `eval/test_context_overflow.py` — it runs the
anchor-fill-recall test against both baseline and ContextCore modes via the
HTTP API.

---

## 10. New file: `.env` (create at repo root — NOT committed)

```
WANDB_API_KEY=<your_wandb_api_key>
```

Already in `.gitignore`. Must be created manually after every fresh clone.

---

## How to start the server

```bash
npm install
node --require ./setup.cjs src/server.js
```

## How to run the context overflow test

In a second terminal (with the server running):
```bash
pip install requests
python eval/test_context_overflow.py
```

---

## Why these changes

| Original | Replacement | Reason |
|---|---|---|
| `NVIDIA_API_KEY` + `https://integrate.api.nvidia.com/v1` | `WANDB_API_KEY` + `https://api.inference.wandb.ai/v1` | Only WandB API key available |
| `google/gemma-3n-e4b-it` | `OpenPipe/Qwen3-14B-Instruct` | Not available on WandB inference |
| `nvidia/nv-embed-v1` (API embedding) | Local hash-based embedding | WandB has no embedding endpoint |
| Score threshold `0.6` | `0.2` | Hash embeddings produce lower cosine scores than neural embeddings |
| Direct `node src/server.js` | `node --require ./setup.cjs src/server.js` | Node 18.15 `File` global polyfill needed |
