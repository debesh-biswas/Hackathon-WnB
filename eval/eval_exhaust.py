"""
ContextCore Context Exhaustion Test
Plants facts, then burns tokens with long-response filler until baseline
approaches gemma-3n-e4b-it's 33K token context window, then tests recall.

Run AFTER starting the server: node src/server.js
Usage: python eval/eval_exhaust.py
"""
import wandb
import weave
import requests
import uuid
import sys

SERVER = "http://localhost:3000"
CONTEXT_WINDOW = 33000  # gemma-3n-e4b-it token limit

SEED_FACTS = [
    "My secret project codename is NIGHTHAWK.",
    "The production database password prefix is XKCD-",
    "My team lead's name is Priya Mehta.",
]

# Each prompt is designed to elicit a long response — burns tokens fast
FILLER_PROMPTS = [
    "Write a detailed explanation of how TCP/IP networking works, including all layers.",
    "Explain the entire history of the Roman Empire from founding to fall in detail.",
    "Write a complete tutorial on async/await in JavaScript with many code examples.",
    "Explain how relational databases work internally, including B-tree indexes.",
    "Write a detailed guide on Docker and containerization from scratch.",
    "Explain machine learning from first principles with mathematical notation.",
    "Write a comprehensive guide to REST API design best practices with examples.",
    "Explain how the Linux kernel manages memory, including virtual memory and paging.",
    "Write a detailed history of programming languages from FORTRAN to today.",
    "Explain how HTTPS and TLS work in complete detail, including the handshake.",
    "Write a comprehensive overview of distributed systems and consensus algorithms.",
    "Explain how compilers work, from lexing to code generation, with examples.",
    "Write a complete guide to SQL query optimization and execution plans.",
    "Explain cryptography fundamentals: symmetric, asymmetric, hashing, and signing.",
    "Write a detailed tutorial on React hooks with extensive code examples.",
    "Explain how GPUs work and why they are fast for matrix operations.",
    "Write a guide to Kubernetes architecture and all its core components.",
    "Explain how version control systems like Git work internally.",
    "Write a comprehensive overview of the CAP theorem and distributed databases.",
    "Explain how operating system scheduling algorithms work in detail.",
]

RECALL_QUESTIONS = [
    {"question": "What is the secret project codename?",             "expected": "nighthawk"},
    {"question": "What is the production database password prefix?", "expected": "xkcd"},
    {"question": "What is my team lead's name?",                     "expected": "priya"},
]


@weave.op()
def send_message(message: str, session_id: str, use_context_core: bool = True):
    try:
        resp = requests.post(f"{SERVER}/chat", json={
            "message": message,
            "sessionId": session_id,
            "useContextCore": use_context_core,
        }, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        tok = data.get("tokenEstimate", 0)
        data["context_window_tokens"] = tok
        data["context_window_pct"]    = round(tok / CONTEXT_WINDOW * 100, 1)
        return data
    except requests.HTTPError as e:
        print(f"  [HTTP error] {e}")
        return {"tokenEstimate": 0, "context_window_tokens": 0, "context_window_pct": 0.0, "response": "[ERROR: likely context overflow]"}


def run_exhaust(mode="contextcore"):
    use_cc = (mode == "contextcore")
    session_id = str(uuid.uuid4())

    run = wandb.init(
        project="contextcore-exhaust",
        name=f"exhaust-{mode}",
        config={
            "mode": mode,
            "context_window": CONTEXT_WINDOW,
            "n_seed_facts": len(SEED_FACTS),
            "n_filler_prompts": len(FILLER_PROMPTS),
        },
    )

    cumulative_tokens = 0
    turn = 0

    # Phase 1: Plant facts
    print(f"\n[{mode}] Planting {len(SEED_FACTS)} seed facts...")
    for fact in SEED_FACTS:
        result = send_message(fact, session_id, use_context_core=use_cc)
        tok = result.get("tokenEstimate", 0)
        cumulative_tokens += tok
        turn += 1
        pct = cumulative_tokens / CONTEXT_WINDOW * 100
        print(f"  Turn {turn}: seeded. tokens={tok}, cumulative={cumulative_tokens} ({pct:.1f}%)")
        wandb.log({
            "turn": turn,
            "token_estimate": tok,
            "cumulative_tokens": cumulative_tokens,
            "pct_context_used": pct,
            "phase": "seed",
        })

    # Phase 2: Burn tokens with verbose filler
    print(f"[{mode}] Burning tokens with {len(FILLER_PROMPTS)} verbose filler turns...")
    for prompt in FILLER_PROMPTS:
        result = send_message(prompt, session_id, use_context_core=use_cc)
        tok = result.get("tokenEstimate", 0)
        cumulative_tokens += tok
        turn += 1
        pct = cumulative_tokens / CONTEXT_WINDOW * 100
        print(f"  Turn {turn}: filler. tokens={tok}, cumulative={cumulative_tokens} ({pct:.1f}% of context)")
        wandb.log({
            "turn": turn,
            "token_estimate": tok,
            "cumulative_tokens": cumulative_tokens,
            "pct_context_used": pct,
            "phase": "filler",
        })
        # Stop early once baseline is clearly overflowing
        if not use_cc and cumulative_tokens >= CONTEXT_WINDOW * 0.9:
            print(f"  [{mode}] Context at {pct:.1f}% — stopping filler, proceeding to recall.")
            break

    # Phase 3: Recall questions
    print(f"[{mode}] Asking {len(RECALL_QUESTIONS)} recall questions (context at {cumulative_tokens / CONTEXT_WINDOW * 100:.1f}%)...")
    scores = []
    for q in RECALL_QUESTIONS:
        result = send_message(q["question"], session_id, use_context_core=use_cc)
        response_text = result.get("response", "")
        tok = result.get("tokenEstimate", 0)
        cumulative_tokens += tok
        turn += 1
        passed = q["expected"].lower() in response_text.lower()
        scores.append(int(passed))
        pct = cumulative_tokens / CONTEXT_WINDOW * 100

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] Q: {q['question']}")
        if not passed:
            print(f"         Response: {response_text[:120]}...")

        wandb.log({
            "turn": turn,
            "token_estimate": tok,
            "cumulative_tokens": cumulative_tokens,
            "pct_context_used": pct,
            "phase": "recall",
            "question": q["question"],
            "expected": q["expected"],
            "response": response_text,
            "passed": passed,
        })

    accuracy = sum(scores) / len(scores) if scores else 0.0
    wandb.log({"recall_accuracy": accuracy, "final_cumulative_tokens": cumulative_tokens})
    wandb.finish()
    print(f"[{mode}] Recall accuracy: {accuracy:.0%} | turns: {turn} | est. tokens: {cumulative_tokens:,}")
    return accuracy


if __name__ == "__main__":
    print("ContextCore Context Exhaustion Test")
    print("=" * 50)
    print(f"Context window: {CONTEXT_WINDOW:,} tokens (gemma-3n-e4b-it)")
    print()

    weave.init("contextcore-exhaust")

    try:
        requests.get(f"{SERVER}/memories", timeout=5)
    except Exception:
        print("ERROR: Server is not running. Start it with: node src/server.js")
        sys.exit(1)

    # Run baseline first (vector store empty = fair test)
    baseline_acc = run_exhaust("baseline")
    # Then ContextCore (same facts seeded via memory pipeline)
    contextcore_acc = run_exhaust("contextcore")

    print("\n" + "=" * 50)
    print("RESULTS:")
    print(f"  Baseline:    {baseline_acc:.0%}  (context overflow → forgets)")
    print(f"  ContextCore: {contextcore_acc:.0%}  (vector memory → recalls)")
    print(f"  Delta:       +{(contextcore_acc - baseline_acc):.0%}")
