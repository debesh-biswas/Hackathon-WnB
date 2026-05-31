export function packContext(memories, recentTurns, currentMessage) {
  const memoryBlock = memories.length > 0
    ? `## Relevant background (only use if directly related to the question):\n${memories.map(m => `- ${m.text}`).join('\n')}`
    : '';

  const historyBlock = recentTurns.length > 0
    ? `## Recent conversation:\n${recentTurns.map(t => `${t.role}: ${t.content}`).join('\n')}`
    : '';

  const memoryInstruction = memories.length > 0
    ? 'Use the background facts below only if the question directly requires them. Never mention that you are recalling anything.'
    : '';

  const parts = [
    'You are a helpful assistant. Answer only what the user asked — nothing more, nothing less.',
    'Match your response length to the question: technical questions deserve thorough answers, simple ones get short ones.',
    'Never append unsolicited context, memory references, follow-up notes, or information about previous topics.',
    memoryInstruction,
    '',
    memoryBlock,
    historyBlock,
  ].filter(Boolean);
  const systemPrompt = parts.join('\n').trim();

  return {
    system: systemPrompt,
    messages: [{ role: 'user', content: currentMessage }],
    memoriesUsed: memories,
    tokenEstimate: Math.round((systemPrompt.length + currentMessage.length) / 4),
  };
}
