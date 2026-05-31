import { runIntake } from './agents/intake.js';
import { runMemoryRead } from './agents/memoryRead.js';
import { packContext } from './agents/contextPacker.js';
import { runGrounding } from './agents/grounding.js';
import { runMemoryWrite } from './agents/memoryWrite.js';

export async function runPipeline(userMessage, sessionHistory, store, client) {
  const startTime = Date.now();

  // Step 1: Intake
  const intakeStart = Date.now();
  const intake = await runIntake(userMessage, client);
  const intake_ms = Date.now() - intakeStart;

  // Step 2: Memory Read
  const memReadStart = Date.now();
  const memories = await runMemoryRead(userMessage, store);
  const memRead_ms = Date.now() - memReadStart;

  // Step 3: Pack context (last 6 turns = 3 exchanges)
  const recentTurns = sessionHistory.slice(-6);
  const packed = packContext(memories, recentTurns, userMessage);

  // Step 4: Main LLM call
  const llmStart = Date.now();
  const llmRes = await client.chat.completions.create({
    model: 'google/gemma-3n-e4b-it',
    max_tokens: 1000,
    messages: [
      { role: 'system', content: packed.system },
      ...packed.messages,
    ],
  });
  const rawResponse = llmRes.choices[0].message.content;
  const llm_ms = Date.now() - llmStart;

  // Step 5: Grounding check
  const groundStart = Date.now();
  const grounded = await runGrounding(rawResponse, memories, client);
  const ground_ms = Date.now() - groundStart;

  // Step 6: Memory Write — ASYNC, does not block response
  runMemoryWrite(userMessage, grounded.response, client, store).catch(console.error);

  return {
    response: grounded.response,
    groundingPassed: grounded.passed,
    contradictions: grounded.contradictions,
    memoriesUsed: packed.memoriesUsed,
    tokenEstimate: packed.tokenEstimate,
    intake,
    timing: {
      intake_ms,
      memRead_ms,
      llm_ms,
      ground_ms,
      total_ms: Date.now() - startTime,
    },
  };
}
