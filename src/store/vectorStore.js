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
