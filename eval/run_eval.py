"""Evaluate retrieval (v1 baseline vs BM25 vs Dense vs Hybrid, Japanese and English queries),
the answer verifier (easy and hard number errors) and the agent.

    python -m eval.run_eval --offline                  # BM25 + rule check only, no API key needed
    python -m eval.run_eval                            # + Dense / Hybrid retrieval + LLM judge
    python -m eval.run_eval --e2e                      # + full agent answers (≈5 LLM calls / question)
    python -m eval.run_eval --e2e --update-readme      # also rewrite the results block in both READMEs
    python -m eval.run_eval --offline --min-hit5 0.9   # CI quality gate: exit 1 on regression

Every number in README_EN.md / README.md comes from this script: the block between
<!-- EVAL:START --> and <!-- EVAL:END --> is generated (English / Japanese labels), never typed by hand.
The full report (with per-question results) is written to eval/results.md.
"""

import argparse
import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from langchain_core.callbacks import UsageMetadataCallbackHandler
from rank_bm25 import BM25Okapi

from app.config import ROOT, get_settings
from app.ingest import load_corpus
from app.retriever import HybridRetriever
from app.tokenizer import normalize, tokenize
from eval import fault_injection

GOLDEN = Path(__file__).with_name("golden_set.jsonl")
REPORT = Path(__file__).with_name("results.md")
READMES = {"en": ROOT / "README_EN.md", "ja": ROOT / "README.md"}
MARK_START, MARK_END = "<!-- EVAL:START -->", "<!-- EVAL:END -->"
K = 5

# Display text for the generated tables. Keys are the same in both languages.
TEXT = {
    "en": {
        "modes": {
            "bm25": "BM25 (page chunks)",
            "dense": "Dense (embeddings)",
            "hybrid": "**Hybrid (BM25 + Dense, RRF)**",
        },
        "verifiers": {
            "LLM judge": "LLM judge only",
            "Rule-based number check": "Rule-based number check only",
            "LLM judge + rule check (deployed)": "**LLM judge + number check (deployed)**",
            "Rule-based number check (hard)": "Rule-based number check only · hard",
            "LLM judge (hard)": "LLM judge only · hard",
            "LLM judge + rule check (deployed) (hard)": "**LLM judge + number check (deployed)** · hard",
        },
        "retriever": "Retriever",
        "v1": "v1 (whole-document BM25)",
        "verifier": "Verifier",
        "false_alarm": "False alarm",
        "metric": "Metric",
        "value": "Value",
        "e2e": [
            "Answer accuracy (expected fact present)",
            "Citation precision (cites the correct document)",
            "Refusal accuracy (unanswerable questions)",
            "Inspection pass rate (all 4 checks passed)",
            "Latency p50 / p95",
            "Avg. tokens per question",
        ],
        "info": "`{llm}` · `{emb}` · {chunks} chunks · {n} questions · {day}",
        "h_retrieval": "**Retrieval** (correct document and page)",
        "h_retrieval_en": "**Cross-lingual retrieval** (the same {n} questions typed in English, searched as-is, "
        "without the agent's translation step)",
        "h_verifier": "**Number-error detection** — easy: {s} sentences with one digit changed · "
        "hard: {h} sentences where the number is swapped for another number from the same page",
        "h_e2e": "**Full agent**",
        "offline_note": "_Hybrid search, LLM judge and full-agent results appear after running with an API key._",
    },
    "ja": {
        "modes": {
            "bm25": "BM25（ページ単位）",
            "dense": "Dense（Embedding）",
            "hybrid": "**Hybrid（BM25 + Dense, RRF）**",
        },
        "verifiers": {
            "LLM judge": "LLM判定のみ",
            "Rule-based number check": "ルールによる数値照合のみ",
            "LLM judge + rule check (deployed)": "**LLM判定 + 数値照合（本番構成）**",
            "Rule-based number check (hard)": "ルールによる数値照合のみ・高難度",
            "LLM judge (hard)": "LLM判定のみ・高難度",
            "LLM judge + rule check (deployed) (hard)": "**LLM判定 + 数値照合（本番構成）**・高難度",
        },
        "retriever": "検索方式",
        "v1": "v1（文書単位BM25）",
        "verifier": "検証方式",
        "false_alarm": "誤検知率",
        "metric": "指標",
        "value": "値",
        "e2e": [
            "回答正解率（期待する事実を含む）",
            "出典の適合率（正しい文書を引用）",
            "回答拒否の正解率（回答不能な質問）",
            "検品合格率（4項目すべて合格）",
            "レイテンシ p50 / p95",
            "平均トークン数 / 質問",
        ],
        "info": "`{llm}` · `{emb}` · {chunks} チャンク · 質問 {n} 件 · {day}",
        "h_retrieval": "**検索精度**（正しい文書とページ）",
        "h_retrieval_en": "**クロスリンガル検索**（同じ質問 {n} 件を英語で入力し、エージェントの日本語変換を通さず"
        "そのまま検索）",
        "h_verifier": "**数値誤りの検出** — 標準：数字を1桁変えた文 {s} 件 · "
        "高難度：同じページ内の別の数字に差し替えた文 {h} 件",
        "h_e2e": "**エージェント全体**",
        "offline_note": "_Hybrid検索・LLM判定・エージェント全体の結果は、APIキーを設定して実行すると追加されます。_",
    },
}


def _norm(text: str) -> str:
    """Same normalisation for expected facts and answers: "1万7,129" == "1万7129"."""
    return normalize(text).replace(",", "")


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
def v1_doc_hit_at_1(chunks, cases: list[dict]) -> float:
    """v1 baseline: BM25 over whole documents (one entry per PDF), as in the first version.
    It can only say *which document*, never *which page*, so only DocHit@1 is reported."""
    doc_ids = sorted({c.doc_id for c in chunks})
    texts = {d: "\n".join(c.text for c in chunks if c.doc_id == d) for d in doc_ids}
    bm25 = BM25Okapi([tokenize(texts[d]) for d in doc_ids])
    hits = sum(doc_ids[int(np.argmax(bm25.get_scores(tokenize(c["question"]))))] == c["doc_id"] for c in cases)
    return hits / len(cases)


def retrieval_metrics(retriever, cases: list[dict], mode: str, field: str = "question") -> dict:
    doc_hit1 = hit1 = hitk = rr = 0.0
    for case in cases:
        hits = retriever.search(case[field], k=K, mode=mode)
        rank = next(
            (i for i, h in enumerate(hits, 1) if h.chunk.doc_id == case["doc_id"] and h.chunk.page in case["pages"]),
            None,
        )
        doc_hit1 += bool(hits) and hits[0].chunk.doc_id == case["doc_id"]
        hit1 += rank == 1
        hitk += rank is not None
        rr += 1 / rank if rank else 0
    n = len(cases)
    return {"mode": mode, "DocHit@1": doc_hit1 / n, "Hit@1": hit1 / n, f"Hit@{K}": hitk / n, f"MRR@{K}": rr / n}


# --------------------------------------------------------------------------- #
# End-to-end agent
# --------------------------------------------------------------------------- #
def e2e_metrics(graph, cases: list[dict]) -> tuple[dict, list[dict]]:
    rows, latencies, tokens = [], [], []
    for case in cases:
        usage = UsageMetadataCallbackHandler()
        t0 = time.perf_counter()
        state = graph.invoke({"question": case["question"], "trace": []}, {"callbacks": [usage]})
        latencies.append(time.perf_counter() - t0)
        tokens.append(sum(u.get("total_tokens", 0) for u in usage.usage_metadata.values()))

        context = state.get("context") or []
        cited = [context[i - 1] for i in state.get("cited_ids") or []]
        answer = _norm(state.get("answer", ""))
        if case["doc_id"]:
            correct = all(_norm(k) in answer for k in case["must_include"])
            cite_ok = [c.doc_id == case["doc_id"] for c in cited]
            precision = sum(cite_ok) / len(cite_ok) if cite_ok else 0.0
        else:  # unanswerable question → the correct behaviour is to refuse
            correct = state.get("status") == "insufficient_evidence"
            precision = None
        failed = [c["name"] for c in state.get("checks") or [] if not c["passed"]]
        rows.append(
            {
                "id": case["id"],
                "status": state.get("status"),
                "correct": correct,
                "citation_precision": precision,
                "failed_checks": failed,
            }
        )

    answerable = [r for r in rows if r["citation_precision"] is not None]
    negatives = [r for r in rows if r["citation_precision"] is None]
    summary = {
        "answer_accuracy": sum(r["correct"] for r in answerable) / len(answerable),
        "citation_precision": statistics.mean(r["citation_precision"] for r in answerable),
        "refusal_accuracy": sum(r["correct"] for r in negatives) / len(negatives) if negatives else None,
        "inspection_pass_rate": sum(r["status"] == "answered" for r in answerable) / len(answerable),
        "p50_latency_s": statistics.median(latencies),
        "p95_latency_s": float(np.percentile(latencies, 95)),
        "avg_tokens_per_question": statistics.mean(tokens),
    }
    return summary, rows


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def retrieval_table(v1: float | None, metrics: dict[str, dict], t: dict) -> list[str]:
    lines = [f"| {t['retriever']} | DocHit@1 | Hit@1 | Hit@{K} | MRR@{K} |", "|---|---:|---:|---:|---:|"]
    if v1 is not None:
        lines.append(f"| {t['v1']} | {v1:.2f} | – | – | – |")
    for mode, m in metrics.items():
        values = " | ".join(f"{m[key]:.2f}" for key in ("DocHit@1", "Hit@1", f"Hit@{K}", f"MRR@{K}"))
        lines.append(f"| {t['modes'][mode]} | {values} |")
    return lines


def verifier_table(results: dict[str, dict], t: dict) -> list[str]:
    lines = [
        f"| {t['verifier']} | Precision | Recall | F1 | {t['false_alarm']} | TP / FP / FN / TN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, name in t["verifiers"].items():  # fixed display order
        if key in results:
            r = results[key]
            lines.append(
                f"| {name} | {r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} "
                f"| {r['false_alarm']:.2f} | {r['tp']} / {r['fp']} / {r['fn']} / {r['tn']} |"
            )
    return lines


def e2e_table(summary: dict, t: dict) -> list[str]:
    s = summary
    refusal = "–" if s["refusal_accuracy"] is None else f"{s['refusal_accuracy']:.2f}"
    values = [
        f"{s['answer_accuracy']:.2f}",
        f"{s['citation_precision']:.2f}",
        refusal,
        f"{s['inspection_pass_rate']:.2f}",
        f"{s['p50_latency_s']:.1f} s / {s['p95_latency_s']:.1f} s",
        f"{s['avg_tokens_per_question']:,.0f}",
    ]
    rows = [f"| {label} | {value} |" for label, value in zip(t["e2e"], values, strict=True)]
    return [f"| {t['metric']} | {t['value']} |", "|---|---:|", *rows]


def results_block(lang: str, r: dict) -> list[str]:
    """The generated README section, in English or Japanese. `r` holds every measured result."""
    t = TEXT[lang]
    block = [
        "> " + t["info"].format(**r["info"]),
        "",
        t["h_retrieval"].format(a=r["info"]["a"]),
        "",
        *retrieval_table(r["v1"], r["retrieval"], t),
        "",
        t["h_retrieval_en"].format(n=r["n_en"]),
        "",
        *retrieval_table(None, r["retrieval_en"], t),
        "",
        t["h_verifier"].format(s=r["samples"], h=r["hard_samples"]),
        "",
        *verifier_table(r["verifiers"], t),
    ]
    if r["summary"] is None:
        if r["offline"]:
            block += ["", t["offline_note"]]
    else:
        block += ["", t["h_e2e"].format(n=r["info"]["n"]), "", *e2e_table(r["summary"], t)]
    return block


def update_readme(path: Path, block: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    if MARK_START not in text or MARK_END not in text:
        sys.exit(f"{path.name} must contain {MARK_START} and {MARK_END}")
    head, rest = text.split(MARK_START, 1)
    _, tail = rest.split(MARK_END, 1)
    path.write_text(f"{head}{MARK_START}\n" + "\n".join(block) + f"\n{MARK_END}{tail}", encoding="utf-8")
    print(f"Results block updated: {path.name}")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="BM25 + rule check only, no OpenAI calls")
    parser.add_argument("--e2e", action="store_true", help="also evaluate full agent answers")
    parser.add_argument("--update-readme", action="store_true", help="rewrite the results block in both READMEs")
    parser.add_argument("--min-hit5", type=float, default=None, help="fail (exit 1) if BM25 Hit@5 is below this")
    args = parser.parse_args()

    settings = get_settings()
    if args.offline:
        chunks, _ = load_corpus(settings)
        retriever, graph, llm, modes = HybridRetriever(chunks, settings), None, None, ("bm25",)
    else:
        from app.main import build_components

        comp = build_components(settings, memory=False)
        retriever, graph, llm, modes = comp.retriever, comp.graph, comp.llm, ("bm25", "dense", "hybrid")

    golden = load_golden()
    answerable = [c for c in golden if c["doc_id"]]

    metrics = {mode: retrieval_metrics(retriever, answerable, mode) for mode in modes}
    metrics_en = {mode: retrieval_metrics(retriever, answerable, mode, field="question_en") for mode in modes}
    v1 = v1_doc_hit_at_1(retriever.chunks, answerable)
    samples = fault_injection.build_samples(retriever.chunks, answerable)
    hard = fault_injection.build_samples(retriever.chunks, answerable, per_case=2, seed=7, hard=True)
    verifiers = fault_injection.run(samples, llm, hard_samples=hard)
    for name, r in {**metrics, **{f"{m}-en": v for m, v in metrics_en.items()}, **verifiers}.items():
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})

    summary, rows = (None, [])
    if args.e2e and graph is not None:
        summary, rows = e2e_metrics(graph, golden)
        print(summary)

    cmd = "python -m eval.run_eval" + (" --offline" if args.offline else "") + (" --e2e" if args.e2e else "")
    results = {
        "info": {
            "cmd": cmd,
            "day": f"{date.today():%Y-%m-%d}",
            "llm": settings.chat_model,
            "emb": settings.embedding_model,
            "chunks": len(retriever.chunks),
            "n": len(golden),
            "a": len(answerable),
            "u": len(golden) - len(answerable),
        },
        "v1": v1,
        "retrieval": metrics,
        "retrieval_en": metrics_en,
        "n_en": len(answerable),
        "samples": len(samples),
        "hard_samples": len(hard),
        "verifiers": verifiers,
        "summary": summary,
        "offline": args.offline,
    }

    report = ["# Evaluation report", "", *results_block("en", results)]
    if rows:
        report += ["", "## Per-question results", ""]
        report += ["| id | status | correct | citation precision | failed checks |", "|---|---|---|---|---|"]
        for r in rows:
            precision = "–" if r["citation_precision"] is None else f"{r['citation_precision']:.2f}"
            report.append(
                f"| {r['id']} | {r['status']} | {'✅' if r['correct'] else '❌'} | {precision} "
                f"| {' / '.join(r['failed_checks']) or '–'} |"
            )

    if not args.offline or not REPORT.exists():  # an offline run never overwrites a full report
        REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
        print(f"Report written to {REPORT}")
    if args.update_readme:
        for lang, path in READMES.items():
            update_readme(path, results_block(lang, results))

    hit5 = metrics["bm25"][f"Hit@{K}"]
    if args.min_hit5 is not None and hit5 < args.min_hit5:
        sys.exit(f"Quality gate failed: BM25 Hit@{K} = {hit5:.2f} < {args.min_hit5:.2f}")


if __name__ == "__main__":
    main()
