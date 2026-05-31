/**
 * intake.js
 * Phase 2 — ContextCore
 *
 * Classifies the intent of an incoming user message and extracts named entities.
 *
 * Export: runIntake(message, client) → { intent, entities, hasFact }
 */

const SYSTEM_PROMPT = `Extract structured info from this chat message. Return JSON only, no markdown:
{
  "intent": "question|statement|command|chitchat",
  "entities": ["list of named entities, facts, preferences mentioned"],
  "hasFact": true/false
}`;

/**
 * Strip markdown code fences that Llama models sometimes wrap around JSON output.
 * e.g. ```json\n{...}\n```  →  {...}
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
 * Classify intent and extract entities from a user message.
 *
 * @param {string} message   - The raw user message.
 * @param {import('openai').OpenAI} client - Pre-configured OpenAI-compatible client.
 * @returns {Promise<{ intent: string, entities: string[], hasFact: boolean }>}
 */
export async function runIntake(message, client) {
  const safeDefault = { intent: 'unknown', entities: [], hasFact: false };

  try {
    const res = await client.chat.completions.create({
      model: 'google/gemma-3n-e4b-it',
      max_tokens: 200,
      messages: [
        { role: 'system', content: SYSTEM_PROMPT },
        { role: 'user', content: message },
      ],
    });

    const raw = res.choices[0].message.content;
    const cleaned = stripFences(raw);

    let parsed;
    try {
      parsed = JSON.parse(cleaned);
    } catch {
      return safeDefault;
    }

    return {
      intent: typeof parsed.intent === 'string' ? parsed.intent : 'unknown',
      entities: Array.isArray(parsed.entities) ? parsed.entities : [],
      hasFact: typeof parsed.hasFact === 'boolean' ? parsed.hasFact : false,
    };
  } catch {
    return safeDefault;
  }
}
