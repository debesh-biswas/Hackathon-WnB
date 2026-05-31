"""
Context window overflow test for ContextCore — WandB inference backend.

Runs the anchor-fill-recall methodology against the ContextCore HTTP server
(http://localhost:3000) in two modes:

  BASELINE     (useContextCore=false)
    6-turn sliding window, no memory retrieval.
    Anchor planted at turn 1 scrolls out after 4 exchanges → FAIL.

  CONTEXTCORE  (useContextCore=true)
    Same 6-turn window + vector store retrieval.
    Anchor stored at turn 1, retrieved by similarity → PASS.

Both modes use the same window size — the only variable is the vector store.

Logs per-turn metrics, recall scores, latency, and a comparison table to W&B.

Start the server first:
    cd Hackathon-WnB && node --require ./setup.cjs src/server.js

Then run:
    pip install wandb requests
    python eval/test_context_overflow.py
"""

import json
import time
import uuid
import datetime
import sys
import requests
import wandb
from pathlib import Path

SERVER_URL   = "http://localhost:3000"
FILL_ROUNDS  = 7      # pushes anchor out of the shared 6-turn window
TURN_DELAY   = 3.0    # seconds between turns — avoid rate limits
WANDB_PROJECT = "contextcore-overflow-test"

ANCHOR_FACTS = {
    "codeword":     "AURORA-7734",
    "project":      "Project Nebula",
    "lucky_number": "8291",
}

FILL_TOPICS = [
    "How does TCP/IP congestion control work? Explain slow start and CUBIC.",
    "What are the main differences between REST and GraphQL APIs?",
    "Explain how Docker containers differ from virtual machines.",
    "How does a B-tree index improve database query performance?",
    "What is the CAP theorem and how do distributed databases navigate it?",
    "How does the transformer attention mechanism work at a high level?",
    "What is the difference between optimistic and pessimistic locking?",
]

RECALL_QUESTIONS = {
    "codeword":     "What was my secret codeword? Reply with just the codeword itself.",
    "project":      "What is the name of my project? Reply with just the project name.",
    "lucky_number": "What is my lucky number? Reply with just the number.",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def post_chat(message: str, session_id: str, use_context_core: bool) -> dict:
    resp = requests.post(
        f"{SERVER_URL}/chat",
        json={"message": message, "sessionId": session_id, "useContextCore": use_context_core},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


def reset_memories() -> None:
    requests.delete(f"{SERVER_URL}/memories", timeout=10)
    print("  [memories cleared]")


def get_stored_memories() -> list:
    try:
        return requests.get(f"{SERVER_URL}/memories", timeout=10).json()
    except Exception:
        return []


def anchor_stored(memories: list) -> bool:
    """Check that at least one anchor fact made it into the vector store."""
    for m in memories:
        text = (m.get("text") or "").lower()
        if any(v.lower() in text for v in ANCHOR_FACTS.values()):
            return True
    return False


def score_recall(answers: dict) -> dict:
    return {
        "codeword":     ANCHOR_FACTS["codeword"].lower() in answers["codeword"].lower(),
        "project":      ANCHOR_FACTS["project"].lower() in answers["project"].lower(),
        "lucky_number": ANCHOR_FACTS["lucky_number"] in answers["lucky_number"],
    }


# ── single-mode test run ──────────────────────────────────────────────────────

def run_test(mode_label: str, use_context_core: bool) -> dict:
    session_id  = f"{mode_label}_{uuid.uuid4().hex[:8]}"
    turn_global = 0
    cumulative_tokens = 0

    print(f"\n{'='*62}")
    print(f"  MODE: {mode_label.upper()}  |  session: {session_id}")
    print(f"{'='*62}\n")

    # ── Phase 1: plant anchor facts ──────────────────────────────
    anchor_prompt = (
        "Please remember these three facts for our entire conversation:\n"
        f"1. My secret codeword is {ANCHOR_FACTS['codeword']}\n"
        f"2. I am working on a project called {ANCHOR_FACTS['project']}\n"
        f"3. My lucky number is {ANCHOR_FACTS['lucky_number']}\n\n"
        "Acknowledge each fact explicitly."
    )
    print("Turn 1 — planting anchor facts...")
    result = post_chat(anchor_prompt, session_id, use_context_core)
    tok     = result.get("tokenEstimate", 0)
    timing  = result.get("timing") or {}
    turn_global += 1
    cumulative_tokens += tok

    wandb.log({
        "mode": mode_label, "turn": turn_global, "phase": "anchor",
        "token_estimate": tok, "cumulative_tokens": cumulative_tokens,
        "memories_retrieved": len(result.get("memoriesUsed") or []),
        "latency_total_ms": timing.get("total_ms"),
        "latency_llm_ms":   timing.get("llm_ms"),
    })

    print(f"  Response  : {result['response'][:120]}...")
    print(f"  Tokens    : {tok}  |  latency: {timing.get('total_ms', '—')} ms")
    time.sleep(TURN_DELAY)

    # ── Phase 2: fill rounds ─────────────────────────────────────
    print(f"\nFill rounds ({FILL_ROUNDS}) — pushing anchor out of the 6-turn window...\n")
    for i, topic in enumerate(FILL_TOPICS[:FILL_ROUNDS], start=2):
        print(f"Turn {i} — {topic[:65]}...")
        result = post_chat(topic, session_id, use_context_core)
        tok    = result.get("tokenEstimate", 0)
        timing = result.get("timing") or {}
        turn_global += 1
        cumulative_tokens += tok

        wandb.log({
            "mode": mode_label, "turn": turn_global, "phase": "fill",
            "token_estimate": tok, "cumulative_tokens": cumulative_tokens,
            "memories_retrieved": len(result.get("memoriesUsed") or []),
            "latency_total_ms": timing.get("total_ms"),
            "latency_llm_ms":   timing.get("llm_ms"),
        })

        print(f"  Turn {result.get('turnCount', i-1):>2} | tokens: {tok} | "
              f"latency: {timing.get('total_ms', '—')} ms")
        time.sleep(TURN_DELAY)

    # ── Pre-recall: verify anchor is in the vector store ─────────
    stored = get_stored_memories()
    anchor_in_store = anchor_stored(stored) if use_context_core else None
    if use_context_core:
        status = "✅ found" if anchor_in_store else "❌ NOT found — recall may fail"
        print(f"\n  Vector store check: anchor {status} ({len(stored)} total facts)")
        wandb.log({"mode": mode_label, "anchor_in_store": anchor_in_store,
                   "total_facts_stored": len(stored)})

    # ── Phase 3: recall probes ───────────────────────────────────
    print(f"\nRecall probes — anchor planted {FILL_ROUNDS} turns ago...\n")
    answers = {}
    for key, question in RECALL_QUESTIONS.items():
        result  = post_chat(question, session_id, use_context_core)
        tok     = result.get("tokenEstimate", 0)
        timing  = result.get("timing") or {}
        turn_global += 1
        cumulative_tokens += tok
        answers[key] = result["response"].strip()
        memories_used = result.get("memoriesUsed") or []

        print(f"  Q: {question}")
        print(f"  A: {answers[key]!r}")
        if use_context_core and memories_used:
            print(f"  Retrieved: {len(memories_used)} fact(s) — "
                  + ", ".join(f'"{m["text"][:40]}"' for m in memories_used[:2]))

        passed = ANCHOR_FACTS[key].lower() in answers[key].lower()
        wandb.log({
            "mode": mode_label, "turn": turn_global, "phase": "recall",
            "recall_key": key,
            "expected": ANCHOR_FACTS[key],
            "response": answers[key],
            "passed": passed,
            "token_estimate": tok,
            "cumulative_tokens": cumulative_tokens,
            "memories_retrieved": len(memories_used),
            "latency_total_ms": timing.get("total_ms"),
            "latency_llm_ms":   timing.get("llm_ms"),
        })
        time.sleep(TURN_DELAY)

    scores   = score_recall(answers)
    accuracy = sum(scores.values()) / len(scores)

    wandb.log({
        "mode": mode_label,
        "codeword_pass":     scores["codeword"],
        "project_pass":      scores["project"],
        "lucky_number_pass": scores["lucky_number"],
        "accuracy":          accuracy,
        "total_turns":       turn_global,
        "total_tokens":      cumulative_tokens,
    })

    print(f"\n  Results for {mode_label.upper()}:")
    for key in ("codeword", "project", "lucky_number"):
        status = "PASS" if scores[key] else "FAIL"
        print(f"    {key:>14}: [{status}]  expected={ANCHOR_FACTS[key]!r}  "
              f"got={answers[key][:60]!r}")

    return {
        "mode": mode_label, "session_id": session_id,
        "use_context_core": use_context_core,
        "anchor_in_store": anchor_in_store,
        "fill_rounds": FILL_ROUNDS,
        "anchor_facts": ANCHOR_FACTS,
        "answers": answers, "scores": scores,
        "accuracy": accuracy,
        "total_turns": turn_global,
        "total_tokens": cumulative_tokens,
    }


# ── comparison table ──────────────────────────────────────────────────────────

def log_comparison_table(baseline: dict, contextcore: dict) -> None:
    table = wandb.Table(columns=["Fact", "Expected", "Baseline", "ContextCore"])
    for key in ("codeword", "project", "lucky_number"):
        table.add_data(
            key,
            ANCHOR_FACTS[key],
            "PASS" if baseline["scores"][key]     else "FAIL",
            "PASS" if contextcore["scores"][key]  else "FAIL",
        )
    wandb.log({"recall_comparison": table})

    wandb.log({
        "summary/baseline_accuracy":    baseline["accuracy"],
        "summary/contextcore_accuracy": contextcore["accuracy"],
        "summary/improvement":          contextcore["accuracy"] - baseline["accuracy"],
        "summary/baseline_tokens":      baseline["total_tokens"],
        "summary/contextcore_tokens":   contextcore["total_tokens"],
    })


def print_comparison(baseline: dict, contextcore: dict) -> None:
    print("\n" + "─" * 62)
    print(f"{'FINAL COMPARISON':^62}")
    print("─" * 62)
    hdr = f"  {'Fact':<16}  {'Baseline':^14}  {'ContextCore':^14}"
    print(hdr)
    print("─" * 62)
    for key in ("codeword", "project", "lucky_number"):
        b = "PASS ✓" if baseline["scores"][key]    else "FAIL ✗"
        c = "PASS ✓" if contextcore["scores"][key] else "FAIL ✗"
        print(f"  {key:<16}  {b:^14}  {c:^14}")
    print("─" * 62)
    print(f"  {'ACCURACY':<16}  {baseline['accuracy']*100:>12.0f}%  "
          f"{contextcore['accuracy']*100:>12.0f}%")
    print(f"  {'TOTAL TOKENS':<16}  {baseline['total_tokens']:>12,}  "
          f"{contextcore['total_tokens']:>12,}")
    print("─" * 62)
    improvement = contextcore["accuracy"] - baseline["accuracy"]
    if improvement > 0:
        print(f"\n  ContextCore outperformed baseline by {improvement*100:.0f} pp.")
    elif improvement == 0:
        print("\n  Both modes scored the same.")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        requests.get(f"{SERVER_URL}/memories", timeout=5).raise_for_status()
    except Exception:
        print(f"\nERROR: Cannot reach server at {SERVER_URL}")
        print("Start it:  node --require ./setup.cjs src/server.js\n")
        sys.exit(1)

    run = wandb.init(
        project=WANDB_PROJECT,
        name=f"overflow-test-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "fill_rounds":   FILL_ROUNDS,
            "window_size":   6,
            "anchor_facts":  ANCHOR_FACTS,
            "model":         "OpenPipe/Qwen3-14B-Instruct",
            "server":        SERVER_URL,
        },
    )

    print(f"\nContextCore context overflow test  |  W&B run: {run.name}")
    print(f"Fill rounds : {FILL_ROUNDS}")
    print(f"Window size : 6 turns for BOTH modes — only difference is vector store retrieval")
    print(f"Anchor scrolls out of the shared window after turn 4.\n")

    reset_memories()
    baseline = run_test("baseline", use_context_core=False)

    reset_memories()
    time.sleep(2)
    contextcore = run_test("contextcore", use_context_core=True)

    print_comparison(baseline, contextcore)
    log_comparison_table(baseline, contextcore)
    wandb.finish()

    # Save local JSON
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"contextcore_overflow_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "wandb_run": run.name,
                   "fill_rounds": FILL_ROUNDS, "anchor_facts": ANCHOR_FACTS,
                   "baseline": baseline, "contextcore": contextcore},
                  f, indent=2, ensure_ascii=False)
    print(f"Results saved → {path}")
    print(f"W&B dashboard → {run.url}")


if __name__ == "__main__":
    main()
