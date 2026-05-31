import 'dotenv/config';
import https from 'https';

const KEY = process.env.WANDB_API_KEY;
console.log('API Key loaded:', KEY ? `${KEY.slice(0, 10)}...` : 'MISSING');

const body = JSON.stringify({
  model: 'OpenPipe/Qwen3-14B-Instruct',
  max_tokens: 10,
  messages: [{ role: 'user', content: 'Say ok' }],
});

const options = {
  hostname: 'api.inference.wandb.ai',
  path: '/v1/chat/completions',
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${KEY}`,
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
  },
};

console.log('Connecting to api.inference.wandb.ai...');

const req = https.request(options, (res) => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    console.log('HTTP Status:', res.statusCode);
    console.log('Response:', data.slice(0, 300));
  });
});

req.on('error', (e) => {
  console.log('Network error:', e.code, e.message);
});

req.setTimeout(10000, () => {
  console.log('Timed out after 10s');
  req.destroy();
});

req.write(body);
req.end();
