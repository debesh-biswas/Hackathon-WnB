/**
 * memoryWrite.js
 * Phase 2 — ContextCore
 *
 * After each conversation turn, extracts 2–5 atomic facts and persists them
 * to the vector store.
 *
 * Export: runMemoryWrite(userMsg, assistantMsg, client, store) → string[]
 */

const SYSTEM_PROMPT = `Given a conversation turn, extract 2–5 atomic facts worth remembering long-term.
Facts should be: specific, self-contained, and useful for future context.
Return a JSON array only, no markdown: ["fact 1", "fact 2", ...]
Skip pleasantries, greetings, and non-informational content.

Good examples:
- "User's name is Aryan"
- "User is building a B2B SaaS product"
- "User's team uses TypeScript and PostgreSQL"
- "User fixed a race condition bug in their auth service last week"`;

/**
 * Strip markdown code fences that Llama models sometimes wrap around JSON output.
 *
 * @param {string} text
 * @returns {string}
 */
function stripFences(text) {
  return text
    .replace(/^```json\s*/i, '')
    .replace(/```\s*$/i, '')
    .trim();
}

/**
 * Extract memorable facts from a conversation turn and store each one.
 *
 * @param {string} userMsg        - The user's message in this turn.
 * @param {string} assistantMsg   - The assistant's reply in this turn.
 * @param {import('openai').OpenAI} client - Pre-configured OpenAI-compatible client.
 * @param {{ storeFact: (fact: string, meta: object) => Promise<void> }} store
 *   - The vector store instance (must expose `storeFact`).
 * @returns {Promise<string[]>} Array of fact strings that were stored.
 */
export async function runMemoryWrite(userMsg, assistantMsg, client, store) {
  try {
    const userContent = `User said: "${userMsg}"\nAssistant said: "${assistantMsg}"`;

    const res = await client.chat.completions.create({
      model: 'OpenPipe/Qwen3-14B-Instruct',
      max_tokens: 400,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: userContent },
      ],
    });

    const raw = res.choices[0].message.content;
    const cleaned = stripFences(raw);

    let facts;
    try {
      facts = JSON.parse(cleaned);
    } catch {
      return [];
    }

    if (!Array.isArray(facts)) {
      return [];
    }

    // Store each fact; collect only the ones that succeed.
    const stored = [];
    for (const fact of facts) {
      if (typeof fact !== 'string' || fact.trim() === '') continue;
      try {
        await store.storeFact(fact, { source: 'conversation' });
        stored.push(fact);
      } catch (storeErr) {
        console.error('[memoryWrite] Failed to store fact:', storeErr);
      }
    }

    return stored;
  } catch (err) {
    console.error('[memoryWrite] Error during fact extraction:', err);
    return [];
  }
}
