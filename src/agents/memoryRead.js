export async function runMemoryRead(retrievalQuery, store, topK = 5) {
  const candidates = await store.retrieveFacts(retrievalQuery, topK);
  return candidates.filter(m => m.score > 0.6);
}
