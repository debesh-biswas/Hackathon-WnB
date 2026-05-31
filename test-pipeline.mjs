import 'dotenv/config';
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: process.env.WANDB_API_KEY,
  baseURL: 'https://api.inference.wandb.ai/v1',
});

const MODEL = 'OpenPipe/Qwen3-14B-Instruct';

// 1. Test chat
console.log('1. Testing chat model...');
try {
  const res = await client.chat.completions.create({
    model: MODEL,
    max_tokens: 20,
    messages: [{ role: 'user', content: 'Say ok' }],
  });
  console.log('   ✅ Chat OK:', res.choices[0].message.content.trim());
} catch (e) {
  console.log('   ❌ Chat FAILED:', e?.error?.message || e.message);
}

// 2. Test vector store (uses local hash embeddings — no API call)
console.log('2. Testing vector store (local hash embeddings)...');
try {
  const { createStore } = await import('./src/store/vectorStore.js');
  const store = createStore();
  await store.storeFact('test fact about the user');
  const results = await store.retrieveFacts('tell me about the user');
  console.log('   ✅ Vector store OK, retrieved:', results.length, 'results');
} catch (e) {
  console.log('   ❌ Vector store FAILED:', e.message);
}
