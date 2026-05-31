import 'dotenv/config';
import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: process.env.WANDB_API_KEY,
  baseURL: 'https://api.inference.wandb.ai/v1',
});

const MODELS_TO_TRY = [
  'OpenPipe/Qwen3-14B-Instruct',
  'meta-llama/Llama-3.1-8B-Instruct',
  'meta-llama/Llama-3.3-70B-Instruct',
  'microsoft/Phi-4-mini-instruct',
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

console.log('=== Testing WandB Chat Models ===');
for (const m of MODELS_TO_TRY) {
  await testModel(m);
}

console.log('\n(WandB inference has no embedding endpoint — embeddings are handled locally via hash vectors)');
