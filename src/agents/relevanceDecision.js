function stripFences(text) {
  return text.replace(/^```json\s*/i, '').replace(/```\s*$/i, '').trim();
}

const SYSTEM = `You decide which stored facts are relevant to answer a user's question.
Return JSON only:
{
  "useContext": true or false,
  "relevantIndices": [0, 2, ...]
}
useContext=false means none of the facts help answer this question.
relevantIndices lists which facts (0-based index) are actually relevant — omit irrelevant ones.`;

export async function runRelevanceDecision(candidates, message, client) {
  if (candidates.length === 0) return { useContext: false, filteredMemories: [] };
  const factsList = candidates.map((m, i) => `[${i}] ${m.text}`).join('\n');
  try {
    const res = await client.chat.completions.create({
      model: 'google/gemma-3n-e4b-it',
      max_tokens: 150,
      messages: [
        { role: 'system', content: SYSTEM },
        { role: 'user', content: `Question: ${message}\n\nFacts:\n${factsList}` },
      ],
    });
    const parsed = JSON.parse(stripFences(res.choices[0].message.content));
    if (!parsed.useContext || !Array.isArray(parsed.relevantIndices) || parsed.relevantIndices.length === 0) {
      return { useContext: false, filteredMemories: [] };
    }
    const filtered = parsed.relevantIndices
      .filter(i => typeof i === 'number' && i >= 0 && i < candidates.length)
      .map(i => candidates[i]);
    return { useContext: filtered.length > 0, filteredMemories: filtered };
  } catch {
    return { useContext: true, filteredMemories: candidates };
  }
}
