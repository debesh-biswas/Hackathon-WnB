"""
ContextCore Memory Quality Evaluation
Run AFTER starting the server: node src/server.js
Usage: python eval/eval.py
"""
import wandb
import weave
import requests
import uuid
import sys

CONTEXT_WINDOW = 33000  # gemma-3n-e4b-it token limit

SERVER = "http://localhost:3000"

SETUP_MESSAGES = [
    "My name is Aryan. I'm building a B2B SaaS product.",
    "We use PostgreSQL for our main database and Redis for caching.",
    "My team only writes TypeScript. We never use Python.",
    "Last week I fixed a nasty race condition in our auth service.",
    "Our biggest customer is a fintech company with 50k users.",
]

FILLER_MESSAGES = [
    "Can you explain the trade-offs between SQL and NoSQL databases in detail?",
    "How does async/await work under the hood in JavaScript? Walk me through the event loop.",
    "What are the main differences between REST and GraphQL, and when should I use each?",
    "Explain microservices architecture — benefits, drawbacks, and when it makes sense.",
    "What are the most common security vulnerabilities in web apps and how do I prevent them?",
    "How does database indexing work, and what are the different types of indexes?",
    "Explain the CAP theorem and how it applies to distributed systems design.",
    "What's the difference between authentication and authorisation, and how should I implement both?",
    "How does garbage collection work in modern runtimes like V8?",
    "What are the pros and cons of TypeScript versus plain JavaScript for a growing codebase?",
    "How do I implement rate limiting in a REST API effectively?",
    "Explain the difference between horizontal and vertical scaling with examples.",
    "What is eventual consistency and when is it acceptable in production systems?",
    "How does a content delivery network work and when should I use one?",
    "What are the trade-offs of using an ORM versus writing raw SQL?",
    "Explain how JWT tokens work and what their security implications are.",
    "What is the difference between a message queue and a pub/sub system?",
    "How do I design an API for pagination efficiently at scale?",
    "What are the SOLID principles and why do they matter in practice?",
    "Explain the difference between concurrency and parallelism with real examples.",
]

EVAL_QUESTIONS = [
    {"question": "By the way, what database are we using in our stack?",  "expected": "postgresql"},
    {"question": "And what programming language does my team work in?",   "expected": "typescript"},
    {"question": "What did I fix last week in our auth service?",         "expected": "race condition"},
    {"question": "What kind of company is our biggest customer?",         "expected": "fintech"},
]



@weave.op()
def send_message(message: str, session_id: str, use_context_core: bool = True):
    resp = requests.post(f"{SERVER}/chat", json={
        "message": message,
        "sessionId": session_id,
        "useContextCore": use_context_core,
    }, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    tok = data.get("tokenEstimate", 0)
    data["context_window_tokens"] = tok
    data["context_window_pct"]    = round(tok / CONTEXT_WINDOW * 100, 1)
    return data


def run_eval(mode="contextcore"):
    use_cc = (mode == "contextcore")
    session_id = str(uuid.uuid4())
    scores = []
    cumulative_tokens = 0
    turn_global = 0

    run = wandb.init(
        project="contextcore-demo",
        name=f"eval-{mode}",
        config={"mode": mode, "n_eval_questions": len(EVAL_QUESTIONS)},
    )

    print(f"\n[{mode}] Sending {len(SETUP_MESSAGES)} setup messages...")
    for msg in SETUP_MESSAGES:
        result = send_message(msg, session_id, use_context_core=use_cc)
        tok = result.get("tokenEstimate", 0)
        cumulative_tokens += tok
        turn_global += 1
        wandb.log({
            "turn": turn_global,
            "token_estimate": tok,
            "cumulative_tokens": cumulative_tokens,
            "phase": "setup",
        })

    print(f"[{mode}] Sending {len(FILLER_MESSAGES)} filler messages...")
    for msg in FILLER_MESSAGES:
        result = send_message(msg, session_id, use_context_core=use_cc)
        tok = result.get("tokenEstimate", 0)
        cumulative_tokens += tok
        turn_global += 1
        wandb.log({
            "turn": turn_global,
            "token_estimate": tok,
            "cumulative_tokens": cumulative_tokens,
            "phase": "filler",
        })

    print(f"[{mode}] Running {len(EVAL_QUESTIONS)} eval questions...")
    for i, q in enumerate(EVAL_QUESTIONS):
        result = send_message(q["question"], session_id, use_context_core=use_cc)
        response_text = result.get("response", "")
        tok = result.get("tokenEstimate", 0)
        cumulative_tokens += tok
        turn_global += 1
        passed = q["expected"].lower() in response_text.lower()
        scores.append(int(passed))

        wandb.log({
            "turn": turn_global,
            "token_estimate": tok,
            "cumulative_tokens": cumulative_tokens,
            "phase": "eval",
            "question": q["question"],
            "expected": q["expected"],
            "response": response_text,
            "passed": passed,
            "mode": mode,
        })

        status = "✅" if passed else "❌"
        print(f"  {status} Q: {q['question']}")
        if not passed:
            print(f"     Response: {response_text[:100]}...")

    accuracy = sum(scores) / len(scores)
    wandb.log({"accuracy": accuracy, "mode": mode})
    wandb.finish()
    print(f"[{mode}] Accuracy: {accuracy:.0%}")
    return accuracy


if __name__ == "__main__":
    print("ContextCore Memory Quality Evaluation")
    print("=" * 40)

    weave.init("contextcore-demo")

    # Check server is running
    try:
        requests.get(f"{SERVER}/memories", timeout=5)
    except Exception:
        print("ERROR: Server is not running. Start it with: node src/server.js")
        sys.exit(1)

    baseline_score = run_eval("baseline")
    contextcore_score = run_eval("contextcore")

    print("\n" + "=" * 40)
    print(f"RESULTS:")
    print(f"  Baseline:    {baseline_score:.0%}")
    print(f"  ContextCore: {contextcore_score:.0%}")
    print(f"  Improvement: +{(contextcore_score - baseline_score):.0%}")
