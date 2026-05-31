import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import OpenAI from 'openai';
import { createStore } from './store/vectorStore.js';
import { runPipeline } from './pipeline.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(join(__dirname, '../ui')));

const client = new OpenAI({
  apiKey: process.env.WANDB_API_KEY,
  baseURL: 'https://api.inference.wandb.ai/v1',
});
const store = createStore();
const sessions = {}; // keyed by sessionId, each value is array of { role, content }

// POST /chat
app.post('/chat', async (req, res) => {
  try {
    const { message, sessionId = 'default', useContextCore = true } = req.body;

    if (!message || message.trim() === '') {
      return res.status(400).json({ error: 'message is required and cannot be empty' });
    }

    if (!sessions[sessionId]) {
      sessions[sessionId] = [];
    }

    let result;

    if (useContextCore === true) {
      result = await runPipeline(message, sessions[sessionId], store, client);
    } else {
      // Baseline: 6-turn sliding window, no memory — older turns are forgotten
      const baselineMessages = [
        ...sessions[sessionId].slice(-6),
        { role: 'user', content: message },
      ];
      const BASELINE_SYSTEM = 'You are a helpful assistant. Answer only what the user asked — nothing more, nothing less. Match your response length to the question: technical questions deserve thorough answers, simple ones get short ones. Never append unsolicited context, follow-up notes, or information about previous topics.';
      const totalChars = BASELINE_SYSTEM.length + baselineMessages.reduce((sum, m) => sum + (m.content?.length || 0), 0);
      const raw = await client.chat.completions.create({
        model: 'OpenPipe/Qwen3-14B-Instruct',
        max_tokens: 1000,
        messages: [
          { role: 'system', content: BASELINE_SYSTEM },
          ...baselineMessages,
        ],
      });
      result = {
        response: raw.choices[0].message.content,
        memoriesUsed: [],
        groundingPassed: null,
        contradictions: [],
        tokenEstimate: Math.round(totalChars / 4),
        intake: null,
        timing: null,
      };
    }

    // Push both turns AFTER getting the result
    sessions[sessionId].push({ role: 'user', content: message });
    sessions[sessionId].push({ role: 'assistant', content: result.response });

    return res.json({ ...result, turnCount: sessions[sessionId].length / 2 });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

// GET /memories
app.get('/memories', async (req, res) => {
  try {
    const all = await store.getAllFacts();
    return res.json(all);
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

// DELETE /memories
app.delete('/memories', async (req, res) => {
  try {
    const fs = await import('fs/promises');
    await fs.rm(join(__dirname, '../vector-store'), { recursive: true, force: true });
    return res.json({ ok: true });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

// POST /retrieve
app.post('/retrieve', async (req, res) => {
  try {
    const { query, topK = 5, scoreThreshold = 0.6 } = req.body;
    if (!query?.trim()) return res.status(400).json({ error: 'query is required' });
    const all = await store.retrieveFacts(query, topK);
    const filtered = all.filter(m => m.score > scoreThreshold);
    res.json({ results: filtered, topK, scoreThreshold });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /seed — pre-populate a session with a history array for demo prep
// Body: { sessionId: string, history: [{role, content}] }
app.post('/seed', (req, res) => {
  try {
    const { sessionId, history } = req.body;
    if (!sessionId) return res.status(400).json({ error: 'sessionId is required' });
    if (!Array.isArray(history)) return res.status(400).json({ error: 'history must be an array' });
    sessions[sessionId] = history.map(({ role, content }) => ({ role, content }));
    return res.json({ ok: true, sessionId, turns: sessions[sessionId].length / 2 });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`ContextCore running on http://localhost:${PORT}`));
