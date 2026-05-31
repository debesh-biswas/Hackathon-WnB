export async function runGrounding(response, memories, client) {
  // Short-circuit: nothing to ground against
  if (memories.length === 0) return { passed: true, contradictions: [], response };

  const memoryContext = memories.map(m => `- ${m.text}`).join('\n');

  const res = await client.chat.completions.create({
    model: 'google/gemma-3n-e4b-it',
    max_tokens: 600,
    messages: [
      {
        role: 'system',
        content: `You are a fact-checker. Compare an assistant response against known facts.
Return JSON only, no markdown:
{
  "passed": true,
  "contradictions": [],
  "correctedResponse": null
}
Set passed to false and populate contradictions if the response contradicts any known fact.
Provide correctedResponse only when passed is false; otherwise set it to null.`,
      },
      {
        role: 'user',
        content: `Known facts:\n${memoryContext}\n\nAssistant response to check:\n"${response}"`,
      },
    ],
  });

  // Defensive JSON parsing (strip markdown fences)
  try {
    const text = res.choices[0].message.content.trim();
    const clean = text.replace(/^```json\s*/i, '').replace(/```\s*$/i, '').trim();
    const result = JSON.parse(clean);
    return {
      passed: result.passed ?? true,
      contradictions: result.contradictions || [],
      response: result.correctedResponse || response,
    };
  } catch {
    // On parse error, pass through the original response
    return { passed: true, contradictions: [], response };
  }
}
