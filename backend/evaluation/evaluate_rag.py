"""
RAG Evaluation Suite for Companion AI
======================================
Evaluates the RAG pipeline across 5 key dimensions:

  1. Retrieval Quality   – Were relevant chunks actually fetched?
  2. Context Relevance  – Are retrieved chunks on-topic for the query?
  3. Answer Faithfulness – Is the LLM answer grounded in the retrieved chunks?
  4. Answer Completeness – Does the answer cover the critical aspects of the question?
  5. Latency            – How long does each stage take?

Usage:
  cd backend
  python evaluate_rag.py                      # run with default test questions
  python evaluate_rag.py --questions my.json  # load custom Q&A pairs from a JSON file
  python evaluate_rag.py --no-llm             # skip LLM calls (retrieval-only metrics)
  python evaluate_rag.py --top-k 10           # override retrieval top-k

JSON question file format:
  [
    {
      "question": "How do I fix the cooling problem on my refrigerator?",
      "device_type": "Refrigerator",          // optional filter
      "brand": "Samsung",                      // optional filter
      "model": "RT38K5982SL",                  // optional filter
      "expected_keywords": ["compressor", "thermostat", "coolant"]  // optional
    },
    ...
  ]
"""

import os, sys, json, time, re, argparse, asyncio, textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")  # .env lives in backend/

# ── Lazy imports so we can skip heavy deps when just viewing help ──────────────

def _check_deps():
    missing = []
    for pkg in ["langchain_huggingface", "langchain_qdrant", "qdrant_client"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {missing}\nRun: pip install {' '.join(missing)}")
        sys.exit(1)

# ── ANSI colours (disabled on Windows if no colour support) ───────────────────
try:
    import colorama; colorama.init()
    GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED    = "\033[91m"
    CYAN   = "\033[96m"; BOLD   = "\033[1m";  RESET  = "\033[0m"
except ImportError:
    GREEN = YELLOW = RED = CYAN = BOLD = RESET = ""


# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT TEST QUESTIONS
# These cover common device troubleshooting queries; adjust to match your data.
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "Q1",
        "question": "My refrigerator is not cooling. What should I check first?",
        "device_type": None, "brand": None, "model": None,
        "expected_keywords": ["compressor", "thermostat", "temperature", "cool", "fan"],
    },
    {
        "id": "Q2",
        "question": "The washing machine is making a loud noise during spin cycle. How do I fix it?",
        "device_type": None, "brand": None, "model": None,
        "expected_keywords": ["drum", "bearing", "spin", "balance", "vibration"],
    },
    {
        "id": "Q3",
        "question": "My TV screen is flickering. What are the possible causes?",
        "device_type": None, "brand": None, "model": None,
        "expected_keywords": ["backlight", "signal", "cable", "refresh", "display"],
    },
    {
        "id": "Q4",
        "question": "The microwave is not heating food properly. What could be wrong?",
        "device_type": None, "brand": None, "model": None,
        "expected_keywords": ["magnetron", "power", "turntable", "door", "heat"],
    },
    {
        "id": "Q5",
        "question": "How do I reset the air conditioner to factory settings?",
        "device_type": None, "brand": None, "model": None,
        "expected_keywords": ["reset", "settings", "remote", "power", "button"],
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# METRICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _keyword_hit_rate(text: str, keywords: List[str]) -> float:
    """Fraction of expected keywords found (case-insensitive) in text."""
    if not keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return round(hits / len(keywords), 3)


def _avg_chunk_similarity(chunks: List[Dict[str, Any]]) -> float:
    """Mean relevance score of retrieved chunks."""
    if not chunks:
        return 0.0
    return round(sum(c["relevance_score"] for c in chunks) / len(chunks), 3)


def _faithfulness_heuristic(answer: str, chunks: List[Dict[str, Any]]) -> float:
    """
    Simple faithfulness heuristic:
    Overlap of bigrams between the answer and all retrieved context.
    Range [0, 1]; higher = more grounded in retrieved text.
    """
    if not chunks or not answer:
        return 0.0

    def bigrams(text: str):
        words = re.findall(r"[a-z]+", text.lower())
        return set(zip(words, words[1:]))

    context = " ".join(c["content"] for c in chunks)
    ans_bigs = bigrams(answer)
    ctx_bigs = bigrams(context)
    if not ans_bigs:
        return 0.0
    overlap = len(ans_bigs & ctx_bigs) / len(ans_bigs)
    return round(min(overlap * 2.5, 1.0), 3)   # scale: answers rephrase, so raw overlap is low


def _format_score(score: float, low=0.3, high=0.6) -> str:
    """Colour a score: red < low, yellow < high, green >= high."""
    pct = f"{score:.1%}"
    if score >= high:
        return f"{GREEN}{pct}{RESET}"
    elif score >= low:
        return f"{YELLOW}{pct}{RESET}"
    return f"{RED}{pct}{RESET}"


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class RAGEvaluator:
    def __init__(self, top_k: int = None, skip_llm: bool = False):
        _check_deps()
        self.skip_llm = skip_llm
        self.results: List[Dict[str, Any]] = []

        from app.services.rag_service import rag_service
        self.rag = rag_service

        if top_k is not None:
            # Override k for this eval run
            from app.core.config import settings
            settings.__dict__["retrieval_top_k"] = top_k

    # ──────────────────────────────────────────────────────────────────────────

    async def evaluate_question(self, q: Dict[str, Any]) -> Dict[str, Any]:
        qid      = q.get("id", "?")
        question = q["question"]
        device   = q.get("device_type")
        brand    = q.get("brand")
        model    = q.get("model")
        keywords = q.get("expected_keywords", [])

        print(f"\n{BOLD}{CYAN}[{qid}]{RESET} {question[:80]}")
        if device or brand or model:
            print(f"      Filter → device={device}, brand={brand}, model={model}")

        result: Dict[str, Any] = {
            "id": qid, "question": question,
            "device_type": device, "brand": brand, "model": model,
        }

        # ── 1. Retrieval ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        chunks = await self.rag.retrieve_relevant_chunks(
            query=question, device_type=device, brand=brand, model=model
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000

        num_chunks         = len(chunks)
        avg_score          = _avg_chunk_similarity(chunks)
        context_relevance  = avg_score          # proxy: mean cosine similarity

        # Keyword presence in *retrieved context* (retrieval recall proxy)
        context_text  = " ".join(c["content"] for c in chunks)
        kw_in_context = _keyword_hit_rate(context_text, keywords)

        result.update({
            "num_chunks_retrieved": num_chunks,
            "avg_relevance_score":  avg_score,
            "context_relevance":    context_relevance,
            "keyword_in_context":   kw_in_context,
            "retrieval_ms":         round(retrieval_ms, 1),
            "chunks": chunks,
        })

        print(f"      Chunks: {num_chunks}  |  Avg score: {_format_score(avg_score)}  "
              f"|  Kw-in-context: {_format_score(kw_in_context)}  "
              f"|  Retrieval: {retrieval_ms:.0f}ms")

        # ── 2. Generation (optional) ──────────────────────────────────────────
        if not self.skip_llm and self.rag.llm is not None and chunks:
            t1 = time.perf_counter()
            try:
                gen_result = await self.rag.generate_answer(
                    query=question, device_type=device, brand=brand, model=model, ai_model="groq"
                )
                answer = gen_result.get("answer", "")
            except Exception as e:
                answer = ""
                print(f"      {RED}LLM error: {e}{RESET}")
            generation_ms = (time.perf_counter() - t1) * 1000

            faithfulness  = _faithfulness_heuristic(answer, chunks)
            kw_in_answer  = _keyword_hit_rate(answer, keywords)

            result.update({
                "answer":         answer,
                "faithfulness":   faithfulness,
                "keyword_in_answer": kw_in_answer,
                "generation_ms":  round(generation_ms, 1),
                "total_ms":       round(retrieval_ms + generation_ms, 1),
            })
            print(f"      Faithfulness: {_format_score(faithfulness)}  "
                  f"|  Kw-in-answer: {_format_score(kw_in_answer)}  "
                  f"|  Gen: {generation_ms:.0f}ms")
        elif self.skip_llm:
            result.update({
                "answer": None, "faithfulness": None,
                "keyword_in_answer": None, "generation_ms": None,
                "total_ms": round(retrieval_ms, 1),
            })
        else:
            print(f"      {YELLOW}Skipped LLM (no chunks retrieved or LLM not init){RESET}")
            result.update({
                "answer": None, "faithfulness": None,
                "keyword_in_answer": None, "generation_ms": None,
                "total_ms": round(retrieval_ms, 1),
            })

        return result

    # ──────────────────────────────────────────────────────────────────────────

    async def run(self, questions: List[Dict[str, Any]]):
        print(f"\n{BOLD}{'═'*70}{RESET}")
        print(f"{BOLD}  RAG Evaluation Suite  –  {len(questions)} questions  –  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
        print(f"{BOLD}{'═'*70}{RESET}")

        for q in questions:
            r = await self.evaluate_question(q)
            self.results.append(r)

        self._print_summary()
        self._save_results()

    # ──────────────────────────────────────────────────────────────────────────

    def _print_summary(self):
        print(f"\n{BOLD}{'═'*70}{RESET}")
        print(f"{BOLD}  SUMMARY{RESET}")
        print(f"{BOLD}{'═'*70}{RESET}")

        n = len(self.results)
        if n == 0:
            print("No results.")
            return

        def _mean(key):
            vals = [r[key] for r in self.results if r.get(key) is not None]
            return round(sum(vals) / len(vals), 3) if vals else None

        avg_chunks     = _mean("num_chunks_retrieved")
        avg_rel        = _mean("avg_relevance_score")
        avg_ctx_kw     = _mean("keyword_in_context")
        avg_faith      = _mean("faithfulness")
        avg_ans_kw     = _mean("keyword_in_answer")
        avg_ret_ms     = _mean("retrieval_ms")
        avg_gen_ms     = _mean("generation_ms")
        avg_total_ms   = _mean("total_ms")

        print(f"\n  {'Metric':<35} {'Value':>10}")
        print(f"  {'-'*46}")
        print(f"  {'Questions evaluated':<35} {n:>10}")
        print(f"  {'Avg chunks retrieved':<35} {avg_chunks if avg_chunks else 'N/A':>10}")
        print(f"  {'Avg relevance score':<35} {_format_score(avg_rel) if avg_rel is not None else 'N/A':>20}")
        print(f"  {'Keyword hit rate (context)':<35} {_format_score(avg_ctx_kw) if avg_ctx_kw is not None else 'N/A':>20}")
        if not self.skip_llm:
            print(f"  {'Answer faithfulness':<35} {_format_score(avg_faith) if avg_faith is not None else 'N/A':>20}")
            print(f"  {'Keyword hit rate (answer)':<35} {_format_score(avg_ans_kw) if avg_ans_kw is not None else 'N/A':>20}")
        print(f"  {'Avg retrieval latency (ms)':<35} {avg_ret_ms if avg_ret_ms else 'N/A':>10}")
        if not self.skip_llm and avg_gen_ms is not None:
            print(f"  {'Avg generation latency (ms)':<35} {avg_gen_ms:>10}")
        print(f"  {'Avg total latency (ms)':<35} {avg_total_ms if avg_total_ms else 'N/A':>10}")

        # Per-question table
        print(f"\n{BOLD}  Per-Question Results{RESET}")
        header = f"  {'ID':<5} {'Chunks':>6} {'AvgSim':>7} {'Ctx-Kw':>7}"
        if not self.skip_llm:
            header += f" {'Faith':>7} {'Ans-Kw':>7}"
        header += f" {'Total(ms)':>10}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for r in self.results:
            faith   = r.get("faithfulness")
            ans_kw  = r.get("keyword_in_answer")
            row = (
                f"  {r['id']:<5} "
                f"{r['num_chunks_retrieved']:>6} "
                f"{_format_score(r['avg_relevance_score']):>17} "
                f"{_format_score(r['keyword_in_context']):>17}"
            )
            if not self.skip_llm:
                fs = _format_score(faith) if faith is not None else '   N/A'
                ak = _format_score(ans_kw) if ans_kw is not None else '   N/A'
                row += f" {fs:>17} {ak:>17}"
            row += f" {r.get('total_ms', 'N/A'):>10}"
            print(row)

        # Diagnosis
        print(f"\n{BOLD}  Diagnosis{RESET}")
        if avg_chunks is not None and avg_chunks < 2:
            print(f"  {YELLOW}⚠ Low chunk count — check Qdrant filters, thresholds, or indexed data.{RESET}")
        if avg_rel is not None and avg_rel < 0.4:
            print(f"  {YELLOW}⚠ Low relevance scores — embedding model may not match your domain well.{RESET}")
        if avg_ctx_kw is not None and avg_ctx_kw < 0.4:
            print(f"  {YELLOW}⚠ Keywords missing from retrieved context — retrieval coverage may be poor.{RESET}")
        if avg_faith is not None and avg_faith < 0.3:
            print(f"  {YELLOW}⚠ Low faithfulness — answers may be hallucinated or not grounded in context.{RESET}")
        if avg_total_ms is not None and avg_total_ms > 5000:
            print(f"  {YELLOW}⚠ High latency ({avg_total_ms:.0f}ms) — consider caching embeddings or using a faster LLM.{RESET}")
        if (avg_chunks is None or avg_chunks >= 2) and (avg_rel is None or avg_rel >= 0.4):
            print(f"  {GREEN}✓ Retrieval looks healthy.{RESET}")

    # ──────────────────────────────────────────────────────────────────────────

    def _save_results(self):
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(exist_ok=True)
        out = out_dir / f"rag_eval_{ts}.json"

        # Remove raw chunks (can be large) but keep a summary
        export = []
        for r in self.results:
            row = {k: v for k, v in r.items() if k != "chunks"}
            row["chunk_preview"] = [
                {"content": c["content"][:120], "score": c["relevance_score"]}
                for c in (r.get("chunks") or [])[:3]
            ]
            export.append(row)

        with open(out, "w", encoding="utf-8") as f:
            json.dump({"timestamp": ts, "results": export}, f, indent=2, default=str)

        print(f"\n  {BOLD}Results saved →{RESET} {out}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate the Companion AI RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python evaluate_rag.py
              python evaluate_rag.py --no-llm
              python evaluate_rag.py --questions custom.json --top-k 10
        """)
    )
    p.add_argument("--questions", "-q", default=None,
                   help="Path to a JSON file with question objects (default: built-in set)")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip LLM generation; only evaluate retrieval metrics")
    p.add_argument("--top-k", type=int, default=None,
                   help="Override RETRIEVAL_TOP_K for this run")
    return p.parse_args()


async def main():
    args = parse_args()

    # Load questions
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)
        # Ensure IDs exist
        for i, q in enumerate(questions):
            q.setdefault("id", f"Q{i+1}")
    else:
        questions = DEFAULT_QUESTIONS

    evaluator = RAGEvaluator(top_k=args.top_k, skip_llm=args.no_llm)
    await evaluator.run(questions)


if __name__ == "__main__":
    # sys.path must include the backend root (parent of evaluation/) so app.* imports work
    sys.path.insert(0, str(Path(__file__).parent.parent))
    asyncio.run(main())
