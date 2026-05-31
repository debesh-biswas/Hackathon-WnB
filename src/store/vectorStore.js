import { LocalIndex } from 'vectra';
import OpenAI from 'openai';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __dirname = dirname(fileURLToPath(import.meta.url));

export function createStore() {
  if (!process.env.NVIDIA_API_KEY) {
    throw new Error('NVIDIA_API_KEY environment variable is not set. Get a free key at https://build.nvidia.com');
  }

  const index = new LocalIndex(join(__dirname, '../../vector-store'));

  const client = new OpenAI({
    apiKey: process.env.NVIDIA_API_KEY,
    baseURL: 'https://integrate.api.nvidia.com/v1',
  });

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

  async function embedText(text) {
    const res = await client.embeddings.create({
      model: 'nvidia/nv-embed-v1',
      input: text,
      encoding_format: 'float',
    });
    if (!res.data || !res.data[0]) {
      throw new Error(`Embedding API returned unexpected shape: ${JSON.stringify(res)}`);
    }
    return res.data[0].embedding;
  }

  async function storeFact(fact, metadata = {}) {
    await ensureIndex();
    const vector = await embedText(fact);
    await index.insertItem({
      vector,
      metadata: { text: fact, ts: Date.now(), ...metadata },
    });
  }

  async function retrieveFacts(query, topK = 5) {
    await ensureIndex();
    const vector = await embedText(query);
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
