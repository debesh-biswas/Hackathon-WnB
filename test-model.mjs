import 'dotenv/config';
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: process.env.NVIDIA_API_KEY,
  baseURL: 'https://integrate.api.nvidia.com/v1',
});

const MODELS_TO_TRY = [
  'nvidia/llama-3.1-nemotron-70b-instruct',
  'meta/llama-3.1-70b-instruct',
  'meta/llama-3.1-8b-instruct',
  'mistralai/mistral-7b-instruct-v0.3',
  'meta/llama3-70b-instruct',
  'meta/llama3-8b-instruct',
];

async function testModel(model) {
  try {
    const res = await client.chat.completions.create({
      model,
      max_tokens: 20,
      messages: [{ role: 'user', content: 'Say "ok"' }],
    });
    console.log(`✅ ${model} → "${res.choices[0].message.content.trim()}"`);
    return true;
  } catch (e) {
    const msg = e?.error?.message || e?.message || String(e);
    console.log(`❌ ${model} → ${msg.slice(0, 120)}`);
    return false;
  }
}

async function testEmbedding(model) {
  try {
    const res = await client.embeddings.create({ model, input: 'test', encoding_format: 'float' });
    console.log(`✅ Embedding ${model} → vector length ${res.data[0].embedding.length}`);
    return true;
  } catch (e) {
    const msg = e?.error?.message || e?.message || String(e);
    console.log(`❌ Embedding ${model} → ${msg.slice(0, 120)}`);
    return false;
  }
}

console.log('=== Testing Chat Models ===');
for (const m of MODELS_TO_TRY) {
  await testModel(m);
}

console.log('\n=== Testing Embedding Model ===');
await testEmbedding('nvidia/nv-embedqa-e5-v5');
await testEmbedding('nvidia/nv-embed-v1');
