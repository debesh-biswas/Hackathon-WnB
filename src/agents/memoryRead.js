export async function runMemoryRead(message, store, topK = 5) {
  const candidates = await store.retrieveFacts(message, topK);
  return candidates.filter(m => m.score > 0.6);
}
