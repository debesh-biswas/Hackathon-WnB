"""
ContextCore Retrieval Parameter Tuning
Run AFTER starting the server: node src/server.js
Usage: python eval/retrieval_tune.py

Sweeps topK and score_threshold to find the best retrieval params.
Results logged to W&B.
"""
import wandb
import requests
import uuid
import sys

SERVER = "http://localhost:3000"

# Synthetic facts to seed the vector store
SEED_FACTS = [
    "My name is Aryan. I'm building a B2B SaaS product.",
    "We use PostgreSQL for our main database and Redis for caching.",
    "My team only writes TypeScript. We never use Python.",
    "Last week I fixed a nasty race condition in our auth service.",
    "Our biggest customer is a fintech company with 50k users.",
    "I prefer dark mode in all my development tools.",
    "Our backend runs on AWS using ECS and RDS.",
    "We deploy every Friday using a blue-green deployment strategy.",
    "I started coding when I was 15 years old.",
    "Our API uses REST, not GraphQL.",
]

RETRIEVAL_TEST_CASES = [
    {"query": "What database does the user use?",               "expected_keyword": "postgresql"},
    {"query": "What language does the team code in?",           "expected_keyword": "typescript"},
    {"query": "What cloud provider do they use?",               "expected_keyword": "aws"},
    {"query": "What kind of deployment does the team do?",      "expected_keyword": "blue-green"},
    {"query": "When did the user start programming?",           "expected_keyword": "15"},
]


def send_message(message, session_id, use_context_core=True):
    resp = requests.post(f"{SERVER}/chat", json={
        "message": message,
        "sessionId": session_id,
        "useContextCore": use_context_core,
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_memories():
    resp = requests.get(f"{SERVER}/memories", timeout=10)
    resp.raise_for_status()
    return resp.json()


def reset_memories():
    requests.delete(f"{SERVER}/memories", timeout=10)


def seed_store():
    """Send facts to populate the vector store."""
    session_id = str(uuid.uuid4())
    for fact in SEED_FACTS:
        send_message(fact, session_id, use_context_core=True)
    print(f"Seeded {len(SEED_FACTS)} facts into vector store.")


def run_retrieval_test(query, expected_keyword, top_k, score_threshold):
    """Query /retrieve directly with specific topK and scoreThreshold params."""
    resp = requests.post(f"{SERVER}/retrieve", json={
        "query": query,
        "topK": top_k,
        "scoreThreshold": score_threshold,
    }, timeout=30)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    retrieved_texts = [r.get("text", "").lower() for r in results]
    return any(expected_keyword.lower() in t for t in retrieved_texts)


if __name__ == "__main__":
    print("ContextCore Retrieval Parameter Tuning")
    print("=" * 40)

    try:
        requests.get(f"{SERVER}/memories", timeout=5)
    except Exception:
        print("ERROR: Server is not running. Start it with: node src/server.js")
        sys.exit(1)

    wandb.init(project="contextcore-retrieval-tuning", name="param-sweep")

    # Seed the store once
    reset_memories()
    seed_store()

    best_recall = 0
    best_params = {}

    for top_k in [3, 5, 8]:
        for score_threshold in [0.5, 0.6, 0.65, 0.75, 0.85]:
            hits = 0

            for test in RETRIEVAL_TEST_CASES:
                hit = run_retrieval_test(test["query"], test["expected_keyword"], top_k, score_threshold)
                if hit:
                    hits += 1

            recall = hits / len(RETRIEVAL_TEST_CASES)

            wandb.log({
                "top_k": top_k,
                "score_threshold": score_threshold,
                "recall": recall,
                "hits": hits,
            })

            if recall > best_recall:
                best_recall = recall
                best_params = {"top_k": top_k, "score_threshold": score_threshold}

            print(f"  topK={top_k}, threshold={score_threshold:.2f} → recall={recall:.0%}")

    wandb.finish()

    print("\n" + "=" * 40)
    print(f"BEST PARAMS: topK={best_params.get('top_k')}, threshold={best_params.get('score_threshold')}")
    print(f"Best recall: {best_recall:.0%}")
    print(f"\nUpdate memoryRead.js with: topK={best_params.get('top_k')}, score > {best_params.get('score_threshold')}")
