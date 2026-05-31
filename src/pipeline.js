import { runUnderstanding } from './agents/understanding.js';
import { runMemoryRead } from './agents/memoryRead.js';
import { runRelevanceDecision } from './agents/relevanceDecision.js';
import { packContext } from './agents/contextPacker.js';
import { runGrounding } from './agents/grounding.js';
import { runMemoryWrite } from './agents/memoryWrite.js';

export async function runPipeline(userMessage, sessionHistory, store, client) {
  const startTime = Date.now();
  const recentTurns = sessionHistory.slice(-6);

  // Step 1: Understanding — reformulate query, decide if retrieval is needed
  const understandStart = Date.now();
  const understanding = await runUnderstanding(userMessage, recentTurns, client);
  const understand_ms = Date.now() - understandStart;

  // Step 2: Memory Read — skip entirely if needsContext=false
  const memReadStart = Date.now();
  let candidates = [];
  if (understanding.needsContext) {
    candidates = await runMemoryRead(understanding.retrievalQuery, store);
  }
  const memRead_ms = Date.now() - memReadStart;

  // Step 3: Relevance Decision — skip if no candidates
  const relevanceStart = Date.now();
  let memories = [];
  if (candidates.length > 0) {
    const decision = await runRelevanceDecision(candidates, userMessage, client);
    memories = decision.useContext ? decision.filteredMemories : [];
  }
  const relevance_ms = Date.now() - relevanceStart;

  // Step 4: Pack context
  const packed = packContext(memories, recentTurns, userMessage);

  // Step 5: Main LLM call
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

  // Step 6: Grounding check
  const groundStart = Date.now();
  const grounded = await runGrounding(rawResponse, memories, client);
  const ground_ms = Date.now() - groundStart;

  // Step 7: Memory Write — async, non-blocking
  runMemoryWrite(userMessage, grounded.response, client, store).catch(console.error);

  return {
    response: grounded.response,
    groundingPassed: grounded.passed,
    contradictions: grounded.contradictions,
    memoriesUsed: packed.memoriesUsed,
    tokenEstimate: packed.tokenEstimate,
    intake: {
      intent: understanding.intent,
      retrievalQuery: understanding.retrievalQuery,
      needsContext: understanding.needsContext,
    },
    timing: {
      understand_ms,
      memRead_ms,
      relevance_ms,
      llm_ms,
      ground_ms,
      total_ms: Date.now() - startTime,
    },
  };
}
