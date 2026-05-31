function stripFences(text) {
  return text.replace(/^```json\s*/i, '').replace(/```\s*$/i, '').trim();
}

const SYSTEM = `You help a memory retrieval system decide what to search for.
Given the user's latest message and recent conversation, return JSON only:
{
  "retrievalQuery": "expanded search string — rephrase to surface relevant facts about the user's project, stack, preferences, or past work",
  "needsContext": true or false,
  "intent": "question|statement|command|chitchat"
}
Set needsContext=false ONLY when the question is entirely self-contained with no connection to the user's personal project context (e.g. "explain photosynthesis", "what is 2+2", "tell me a joke").
Set needsContext=true when in doubt.`;

export async function runUnderstanding(message, recentTurns, client) {
  const safeDefault = { retrievalQuery: message, needsContext: true, intent: 'unknown' };
  const historySnippet = recentTurns.slice(-4).map(t => `${t.role}: ${t.content}`).join('\n');
  const userContent = historySnippet
    ? `Recent conversation:\n${historySnippet}\n\nLatest message: ${message}`
    : message;
  try {
    const res = await client.chat.completions.create({
      model: 'google/gemma-3n-e4b-it',
      max_tokens: 200,
      messages: [{ role: 'system', content: SYSTEM }, { role: 'user', content: userContent }],
    });
    const parsed = JSON.parse(stripFences(res.choices[0].message.content));
    return {
      retrievalQuery: typeof parsed.retrievalQuery === 'string' ? parsed.retrievalQuery : message,
      needsContext: parsed.needsContext !== false,
      intent: typeof parsed.intent === 'string' ? parsed.intent : 'unknown',
    };
  } catch {
    return safeDefault;
  }
}
