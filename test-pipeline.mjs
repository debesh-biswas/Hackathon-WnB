import 'dotenv/config';
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: process.env.NVIDIA_API_KEY,
  baseURL: 'https://integrate.api.nvidia.com/v1',
});

const MODEL = 'meta/llama-4-maverick-17b-128e-instruct';

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

// 2. Test embedding
console.log('2. Testing embedding model nvidia/nv-embedqa-e5-v5...');
try {
  const res = await client.embeddings.create({
    model: 'nvidia/nv-embedqa-e5-v5',
    input: 'hello world',
    encoding_format: 'float',
  });
  console.log('   ✅ Embedding OK, vector length:', res.data[0].embedding.length);
} catch (e) {
  console.log('   ❌ Embedding FAILED:', e?.error?.message || e.message);

  // Try fallback
  console.log('   Trying nvidia/nv-embed-v1...');
  try {
    const res2 = await client.embeddings.create({
      model: 'nvidia/nv-embed-v1',
      input: 'hello world',
      encoding_format: 'float',
    });
    console.log('   ✅ nvidia/nv-embed-v1 OK, length:', res2.data[0].embedding.length);
  } catch (e2) {
    console.log('   ❌ nvidia/nv-embed-v1 also FAILED:', e2?.error?.message || e2.message);
  }
}

// 3. Test vector store init
console.log('3. Testing vector store...');
try {
  const { createStore } = await import('./src/store/vectorStore.js');
  const store = createStore();
  await store.storeFact('test fact about the user');
  const results = await store.retrieveFacts('tell me about the user');
  console.log('   ✅ Vector store OK, retrieved:', results.length, 'results');
} catch (e) {
  console.log('   ❌ Vector store FAILED:', e.message);
}
