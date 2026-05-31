import 'dotenv/config';
process.env.NODE_TLS_REJECT_UNAUTHORIZED = '0';
import https from 'https';

const KEY = process.env.NVIDIA_API_KEY;
console.log('API Key loaded:', KEY ? `${KEY.slice(0, 10)}...` : 'MISSING');

const body = JSON.stringify({
  model: 'google/gemma-3n-e4b-it',
  max_tokens: 10,
  messages: [{ role: 'user', content: 'Say ok' }],
});

const options = {
  hostname: 'integrate.api.nvidia.com',
  path: '/v1/chat/completions',
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${KEY}`,
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(body),
  },
};

console.log('Connecting to integrate.api.nvidia.com...');

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
