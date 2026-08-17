"""
Statistical analysis for MediChat RAG evaluation.

Primary question (auto metrics vs LLM-as-a-Judge):
  Spearman's rank correlation on paired per-question scores, with bootstrap
  confidence intervals and Holm–Bonferroni correction across metric pairs.

Secondary (same metric across two backends):
  Assumption-aware paired test (Shapiro on differences → paired t or Wilcoxon).

Why correlation (not a mean-difference test) for auto vs judge:
  Automatic retrieval metrics and judge metrics measure related but different
  constructs on the same questions. The scientifically appropriate question is
  whether they co-vary (agreement / monotonic association), not whether their
  means are equal on a shared [0, 1] scale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats


RETRIEVAL_METRICS = ("hit_rate", "mrr", "source_precision")
JUDGE_METRICS = (
    "faithfulness",
    "answer_relevance",
    "context_relevance",
    "correctness",
)

# Theoretically most aligned pairs for the report (retrieval quality ↔ judge).
DEFAULT_PAIRS = (
    ("source_precision", "context_relevance"),
    ("mrr", "context_relevance"),
    ("source_precision", "correctness"),
    ("mrr", "correctness"),
)


def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric(result: dict, metric: str) -> float:
    if "retrieval_scores" in result and metric in result["retrieval_scores"]:
        return float(result["retrieval_scores"][metric])
    scores = result.get("judge_scores") or result.get("llm_scores")
    if scores and metric in scores:
        score = scores[metric]
        if isinstance(score, dict) and "score" in score:
            return float(score["score"])
        return float(score)
    raise KeyError(f"Metric '{metric}' not found in result entry")


def make_question_map(results):
    mapping = {}
    for item in results:
        q = item.get("question")
        if q is None:
            raise ValueError("All result entries must contain a 'question' field")
        mapping[q] = item
    return mapping


def paired_scores(results_a, results_b, metric):
    map_a = make_question_map(results_a)
    map_b = make_question_map(results_b)
    common = sorted(set(map_a) & set(map_b))
    if not common:
        raise ValueError("No matching questions found between the two result sets")
    x = np.array([extract_metric(map_a[q], metric) for q in common], dtype=float)
    y = np.array([extract_metric(map_b[q], metric) for q in common], dtype=float)
    return x, y, common


def descriptive_univariate(x: np.ndarray) -> dict:
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "n_unique": int(len(np.unique(x))),
    }


def descriptive_paired(x, y):
    diffs = x - y
    return {
        "mean_a": float(np.mean(x)),
        "mean_b": float(np.mean(y)),
        "median_a": float(np.median(x)),
        "median_b": float(np.median(y)),
        "std_a": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
        "std_b": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
        "mean_diff_a_minus_b": float(np.mean(diffs)),
        "median_diff_a_minus_b": float(np.median(diffs)),
    }


def cohens_d_paired(x, y):
    diffs = x - y
    if len(diffs) < 2:
        return None
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(diffs) / sd)


def rank_biserial_from_wilcoxon(x, y):
    diffs = x - y
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    n_pos = int(np.sum(nonzero > 0))
    n_neg = int(np.sum(nonzero < 0))
    return float((n_pos - n_neg) / len(nonzero))


def shapiro_test(values: np.ndarray):
    if len(values) < 3 or len(values) > 5000:
        return None, None
    if float(np.std(values, ddof=1) if len(values) > 1 else 0.0) == 0.0:
        return None, None
    if len(np.unique(values)) < 3:
        return None, None
    stat, p = stats.shapiro(values)
    return float(stat), float(p)


def bootstrap_spearman_ci(x, y, n_boot=5000, alpha=0.05, seed=42):
    """Percentile bootstrap CI for Spearman rho (pairwise complete)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return None, None
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if len(np.unique(xb)) < 2 or len(np.unique(yb)) < 2:
            continue
        rho_b, _ = stats.spearmanr(xb, yb)
        if not np.isnan(rho_b):
            rhos.append(float(rho_b))
    if len(rhos) < max(100, n_boot // 10):
        return None, None
    lo = float(np.percentile(rhos, 100 * (alpha / 2)))
    hi = float(np.percentile(rhos, 100 * (1 - alpha / 2)))
    return lo, hi


def holm_bonferroni(p_values: list[float | None], alpha: float = 0.05):
    """
    Holm–Bonferroni step-down correction.
    Returns list of dicts with adjusted p and reject decision (same order as input).
    None p-values are left as ineligible / not tested.
    """
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None and not np.isnan(p)]
    m = len(indexed)
    out = [
        {
            "p_adjusted": None,
            "significant_holm": False,
            "eligible": p is not None and not (isinstance(p, float) and np.isnan(p)),
        }
        for p in p_values
    ]
    if m == 0:
        return out

    indexed.sort(key=lambda t: t[1])
    adjusted = {}
    reject = {}
    prev_adj = 0.0
    for rank, (i, p) in enumerate(indexed):
        # Holm: compare p_(k) with alpha / (m - k + 1); store adjusted p = min(1, p*(m-k+1))
        adj = min(1.0, p * (m - rank))
        adj = max(adj, prev_adj)  # enforce monotonicity
        prev_adj = adj
        adjusted[i] = adj

    # Step-down reject: once a non-reject occurs, all larger p fail
    can_reject = True
    for rank, (i, p) in enumerate(indexed):
        threshold = alpha / (m - rank)
        if can_reject and p <= threshold:
            reject[i] = True
        else:
            can_reject = False
            reject[i] = False

    for i, p in indexed:
        out[i] = {
            "p_adjusted": float(adjusted[i]),
            "significant_holm": bool(reject[i]),
            "eligible": True,
        }
    return out


def interpret_rho(rho: float | None) -> str:
    if rho is None or (isinstance(rho, float) and np.isnan(rho)):
        return "недефинирана"
    a = abs(rho)
    if a < 0.1:
        strength = "занемарлива"
    elif a < 0.3:
        strength = "слаба"
    elif a < 0.5:
        strength = "умерена"
    elif a < 0.7:
        strength = "силна"
    else:
        strength = "многу силна"
    direction = "позитивна" if rho >= 0 else "негативна"
    return f"{strength} {direction}"


def spearman_pair_analysis(x, y, alpha=0.05, n_boot=5000):
    """Primary association analysis for one auto↔judge pair."""
    base = {
        "n": int(len(x)),
        "descriptives_retrieval": descriptive_univariate(x),
        "descriptives_judge": descriptive_univariate(y),
        "alpha": alpha,
    }

    if len(x) < 3:
        return {
            **base,
            "status": "insufficient_n",
            "message": "Need at least 3 paired observations for Spearman correlation.",
            "spearman_rho": None,
            "spearman_p": None,
            "spearman_ci_95": None,
            "pearson_r": None,
            "pearson_p": None,
        }

    if len(np.unique(x)) < 2:
        return {
            **base,
            "status": "degenerate_retrieval",
            "message": (
                "Retrieval metric has no variance (e.g. hit_rate constantly 1.0). "
                "Correlation is undefined."
            ),
            "spearman_rho": None,
            "spearman_p": None,
            "spearman_ci_95": None,
            "pearson_r": None,
            "pearson_p": None,
        }

    if len(np.unique(y)) < 2:
        return {
            **base,
            "status": "degenerate_judge",
            "message": "Judge metric has no variance on the judged sample. Correlation is undefined.",
            "spearman_rho": None,
            "spearman_p": None,
            "spearman_ci_95": None,
            "pearson_r": None,
            "pearson_p": None,
        }

    rho, p_s = stats.spearmanr(x, y)
    rho, p_s = float(rho), float(p_s)
    ci_lo, ci_hi = bootstrap_spearman_ci(x, y, n_boot=n_boot, alpha=alpha)

    # Secondary: Pearson + normality note (not used for primary inference)
    shapiro_x = shapiro_test(x)
    shapiro_y = shapiro_test(y)
    pearson_ok = (
        shapiro_x[1] is not None
        and shapiro_y[1] is not None
        and shapiro_x[1] > alpha
        and shapiro_y[1] > alpha
    )
    r_p, p_p = stats.pearsonr(x, y)

    note = None
    if len(x) < 20:
        note = (
            f"Sample size n={len(x)} is small; bootstrap CI and p-values are "
            "directionally informative but under-powered. Prefer n≥20–30 (ideally 50)."
        )

    return {
        **base,
        "status": "ok",
        "test_primary": "Spearman rank correlation",
        "null_hypothesis": "H0: ρ = 0 (no monotonic association)",
        "alternative_hypothesis": "H1: ρ ≠ 0",
        "spearman_rho": rho,
        "spearman_p": p_s,
        "spearman_ci_95": (
            {"low": ci_lo, "high": ci_hi} if ci_lo is not None else None
        ),
        "effect_size_interpretation": interpret_rho(rho),
        "significant_uncorrected": bool(p_s < alpha),
        "pearson_r": float(r_p),
        "pearson_p": float(p_p),
        "pearson_assumptions_plausible": pearson_ok,
        "shapiro_retrieval": {"W": shapiro_x[0], "p": shapiro_x[1]},
        "shapiro_judge": {"W": shapiro_y[0], "p": shapiro_y[1]},
        "note": note,
        "message": None,
    }


def entries_with_judge(results, judge_metric):
    judged = []
    for item in results:
        scores = item.get("judge_scores") or item.get("llm_scores")
        if not scores or judge_metric not in scores:
            continue
        judged.append(item)
    return judged


def available_judge_metrics(results) -> list[str]:
    found = set()
    for item in results:
        scores = item.get("judge_scores") or item.get("llm_scores") or {}
        found.update(scores.keys())
    return sorted(found)


def auto_vs_judge_report(
    file_path: Path,
    pairs: list[tuple[str, str]] | None = None,
    alpha: float = 0.05,
    n_boot: int = 5000,
):
    results = load_results(Path(file_path))
    present_judge = available_judge_metrics(results)
    if not present_judge:
        raise ValueError(
            "No judge_scores found in results. Smallest fix:\n"
            "  python -m evaluation.evaluate --augment-results evaluation/results.json "
            "--judge-sample 25 --judge-metrics context_relevance,correctness\n"
            "Then re-run this analysis."
        )

    if pairs is None:
        pairs = [(a, j) for a, j in DEFAULT_PAIRS if j in present_judge]
        # Also include any other available judge metrics with source_precision / mrr
        for j in present_judge:
            for a in ("source_precision", "mrr"):
                if (a, j) not in pairs:
                    pairs.append((a, j))

    pair_results = []
    for retrieval_metric, judge_metric in pairs:
        judged = entries_with_judge(results, judge_metric)
        # Keep only entries that also have the retrieval metric
        usable = []
        for item in judged:
            try:
                extract_metric(item, retrieval_metric)
                usable.append(item)
            except KeyError:
                continue
        if not usable:
            pair_results.append(
                {
                    "retrieval_metric": retrieval_metric,
                    "judge_metric": judge_metric,
                    "status": "missing_data",
                    "message": f"No paired entries for {retrieval_metric} ↔ {judge_metric}",
                    "spearman_p": None,
                }
            )
            continue

        x = np.array([extract_metric(i, retrieval_metric) for i in usable], dtype=float)
        y = np.array([extract_metric(i, judge_metric) for i in usable], dtype=float)
        analysis = spearman_pair_analysis(x, y, alpha=alpha, n_boot=n_boot)
        analysis["retrieval_metric"] = retrieval_metric
        analysis["judge_metric"] = judge_metric
        analysis["n_total_in_file"] = len(results)
        analysis["n_with_judge"] = len(judged)
        pair_results.append(analysis)

    holm = holm_bonferroni([p.get("spearman_p") for p in pair_results], alpha=alpha)
    for pr, h in zip(pair_results, holm):
        pr["holm"] = h
        if h["eligible"] and pr.get("status") == "ok":
            pr["significant_holm"] = h["significant_holm"]
            pr["spearman_p_adjusted"] = h["p_adjusted"]
        else:
            pr["significant_holm"] = False
            pr["spearman_p_adjusted"] = None

    n_judged = max((p.get("n_with_judge") or 0 for p in pair_results), default=0)
    return {
        "analysis": "auto_vs_judge",
        "file": str(file_path),
        "n_total": len(results),
        "n_judged_max": n_judged,
        "judge_metrics_present": present_judge,
        "alpha": alpha,
        "multiple_comparison_correction": "Holm–Bonferroni",
        "primary_test": "Spearman rank correlation (ρ) with bootstrap 95% CI",
        "secondary_test": "Pearson r (reported only; not used for primary inference)",
        "pairs": pair_results,
        "practical_vs_statistical": (
            "Statistical significance (p < α after Holm) indicates evidence against "
            "ρ = 0. Practical significance is judged from |ρ| and the CI: weak |ρ| "
            "may be statistically significant at large n but still imply limited "
            "agreement for replacing human/LLM judgment with automatic metrics alone."
        ),
    }


def summarize_backend_comparison(metric, x, y, alpha=0.05):
    diffs = x - y
    base = {
        "metric": metric,
        "n": len(x),
        "alpha": alpha,
        "descriptives": descriptive_paired(x, y),
    }

    if np.allclose(diffs, 0.0):
        return {
            **base,
            "test": "identical values",
            "shapiro_stat": None,
            "shapiro_p": None,
            "stat": 0.0,
            "p_value": 1.0,
            "significant": False,
            "effect_size": {"cohens_d_paired": 0.0, "rank_biserial": 0.0},
            "ci_mean_diff_95": {"low": 0.0, "high": 0.0},
        }

    shapiro_stat, shapiro_p = shapiro_test(diffs)
    cohens_d = cohens_d_paired(x, y)
    r_rb = rank_biserial_from_wilcoxon(x, y)

    # Bootstrap CI for mean difference (always reported)
    rng = np.random.default_rng(42)
    boot_means = []
    for _ in range(5000):
        idx = rng.integers(0, len(diffs), size=len(diffs))
        boot_means.append(float(np.mean(diffs[idx])))
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    if shapiro_p is not None and shapiro_p > alpha:
        test_name = "paired t-test"
        stat, p_value = stats.ttest_rel(x, y)
        stat, p_value = float(stat), float(p_value)
        if np.isnan(p_value):
            test_name = "Wilcoxon signed-rank test"
            stat, p_value = stats.wilcoxon(x, y, zero_method="wilcox")
            stat, p_value = float(stat), float(p_value)
            primary_effect = {"name": "rank_biserial", "value": r_rb}
        else:
            primary_effect = {"name": "cohens_d_paired", "value": cohens_d}
    else:
        test_name = "Wilcoxon signed-rank test"
        stat, p_value = stats.wilcoxon(x, y, zero_method="wilcox")
        stat, p_value = float(stat), float(p_value)
        primary_effect = {"name": "rank_biserial", "value": r_rb}

    return {
        **base,
        "test": test_name,
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "stat": stat,
        "p_value": p_value,
        "significant": bool(p_value < alpha),
        "effect_size": {
            "primary": primary_effect,
            "cohens_d_paired": cohens_d,
            "rank_biserial": r_rb,
        },
        "ci_mean_diff_95": {"low": ci_lo, "high": ci_hi},
    }


def compare_files(file_a, file_b, metric, alpha=0.05):
    results_a = load_results(Path(file_a))
    results_b = load_results(Path(file_b))
    x, y, questions = paired_scores(results_a, results_b, metric)
    summary = summarize_backend_comparison(metric, x, y, alpha=alpha)
    summary["file_a"] = str(file_a)
    summary["file_b"] = str(file_b)
    summary["questions_compared"] = len(questions)
    return summary


def correlate_metrics(file_path, retrieval_metric, judge_metric, alpha=0.05, n_boot=5000):
    """Single-pair convenience wrapper (same method as auto-vs-judge)."""
    report = auto_vs_judge_report(
        Path(file_path),
        pairs=[(retrieval_metric, judge_metric)],
        alpha=alpha,
        n_boot=n_boot,
    )
    return report["pairs"][0]


def format_human_report(report: dict) -> str:
    lines = [
        "=" * 70,
        "Автоматски метрики ↔ LLM-as-a-Judge — статистичка анализа",
        "=" * 70,
        f"Датотека: {report['file']}",
        f"Вкупно прашања: {report['n_total']} | со judge: {report['n_judged_max']}",
        f"Примарен тест: {report['primary_test']}",
        f"Корекција: {report['multiple_comparison_correction']} (α = {report['alpha']})",
        "",
    ]
    for p in report["pairs"]:
        lines.append("-" * 70)
        lines.append(f"{p.get('retrieval_metric')}  ↔  {p.get('judge_metric')}")
        if p.get("status") != "ok":
            lines.append(f"  Статус: {p.get('status')}")
            lines.append(f"  {p.get('message')}")
            continue
        lines.append(f"  n = {p['n']}")
        lines.append(
            f"  Retrieval: mean={p['descriptives_retrieval']['mean']:.3f}, "
            f"median={p['descriptives_retrieval']['median']:.3f}, "
            f"std={p['descriptives_retrieval']['std']:.3f}"
        )
        lines.append(
            f"  Judge:     mean={p['descriptives_judge']['mean']:.3f}, "
            f"median={p['descriptives_judge']['median']:.3f}, "
            f"std={p['descriptives_judge']['std']:.3f}"
        )
        lines.append(f"  Spearman ρ = {p['spearman_rho']:.4f}")
        lines.append(f"  p (некоригирана) = {p['spearman_p']:.4g}")
        lines.append(f"  p (Holm) = {p.get('spearman_p_adjusted')}")
        if p.get("spearman_ci_95"):
            ci = p["spearman_ci_95"]
            lines.append(f"  95% bootstrap CI за ρ: [{ci['low']:.4f}, {ci['high']:.4f}]")
        lines.append(f"  Ефект (толкување на |ρ|): {p['effect_size_interpretation']}")
        lines.append(
            f"  Значајно по Holm: {'ДА' if p.get('significant_holm') else 'НЕ'}"
        )
        if p.get("note"):
            lines.append(f"  Забелешка: {p['note']}")
    lines.append("-" * 70)
    lines.append(report["practical_vs_statistical"])
    lines.append("=" * 70)
    return "\n".join(lines)


def macedonian_methods_and_results_section(report: dict | None) -> str:
    """Academic-quality Macedonian section for the final report."""
    methods = """## Статистичка споредба на автоматските метрики и LLM-as-a-Judge

### Цел
Целта на статистичката анализа е да се провери дали автоматските метрики за retrieval
(Hit Rate, MRR, Source Precision) се во согласност со оценките добиени преку
LLM-as-a-Judge (на пр. context relevance и correctness) на истите прашања од
евалуациското множество. Оваа споредба е важна затоа што двете семејства метрики
мерат поврзани, но различни аспекти на квалитетот на RAG-системот: автоматските
метрики ја оценуваат прецизноста на пронајдените извори, додека LLM-судијата ја
оценува релевантноста на контекстот и фактичката коректност на генерираниот одговор.

### Избор на статистички тест
Бидејќи за секое прашање постојат парни мерења (автоматска метрика и judge-метрика
на истиот примерок), а прашањето на истражувањето е дали постои монотона поврзаност
меѓу двете оценки, како примарен тест е избран Спирмановиот коефициент на рангова
корелација (Spearman’s ρ). Овој избор е мотивиран од следниве својства на податоците
и дизајнот:

1. мерењата се парни по прашање;
2. метриките се ограничени на интервалот [0, 1] и често се дискретни;
3. дистрибуциите на retrieval-метриките во проектот не се нормални
   (на пр. MRR има мал број различни вредности и силно отстапување од нормалност);
4. Hit Rate во тековните резултати е константен (1.0), па корелацијата за таа
   метрика е методолошки недефинирана и се исклучува од инференција.

Алтернативите се оценети како помалку соодветни за оваа споредба:

- парен t-тест / Wilcoxon signed-rank тест би тестирале еднаквост на средни вредности
  меѓу различни конструкти, што не одговара на прашањето за согласност;
- Mann–Whitney U претпоставува независни примероци, а тука примероците се парни;
- Pearson-овата корелација се пријавува само како секундарна информација, бидејќи
  претпоставките за линеарност и приближна нормалност не се веродостојно исполнети.

За споредба на два backend-а на иста метрика (на пр. MRR), посебно се користи
assumption-aware парен тест (Shapiro–Wilk на разликите → парен t-тест или Wilcoxon).

### Хипотези и ниво на значајност
За секој пар (автоматска метрика, judge-метрика):

- H₀: ρ = 0 (нема монотона асоцијација)
- H₁: ρ ≠ 0 (постои монотона асоцијација)

Нивото на значајност е α = 0.05. Бидејќи се тестираат повеќе парови метрики,
p-вредностите се коригираат со постапката на Holm–Bonferroni. Како големина на ефект
се интерпретира самата |ρ|, заедно со 95% bootstrap интервал на доверба.
Статистичката значајност не се поистоветува со практична значајност: слаба |ρ|
може да биде статистички значајна при поголем n, но сепак да укажува на ограничена
согласност за практична замена на judge-евалуацијата со автоматски метрики.
"""

    if report is None:
        results_block = """
### Резултати
*(Овој дел се пополнува откако ќе се соберат LLM-as-a-Judge оценки на евалуациското
множество и ќе се изврши `python -m evaluation.statistical_analysis auto-vs-judge ...`.
Без judge_scores во резултатските JSON датотеки не се пријавуваат измислени вредности.)*
"""
        return methods + results_block

    lines = [
        "",
        "### Резултати",
        f"Анализата е спроведена на датотеката `{report['file']}` "
        f"(вкупно {report['n_total']} прашања, од кои најмногу "
        f"{report['n_judged_max']} имаат judge-оценки). "
        f"Присутен(и) judge-метрики: {', '.join(report['judge_metrics_present'])}.",
        "",
    ]
    for p in report["pairs"]:
        title = f"**{p.get('retrieval_metric')} ↔ {p.get('judge_metric')}**"
        if p.get("status") != "ok":
            lines.append(f"- {title}: {p.get('message') or p.get('status')}")
            continue
        ci = p.get("spearman_ci_95")
        ci_txt = (
            f"[{ci['low']:.3f}, {ci['high']:.3f}]" if ci else "недостапен"
        )
        holm_p = p.get("spearman_p_adjusted")
        holm_txt = f"{holm_p:.4g}" if holm_p is not None else "н/п"
        sig = "значајна" if p.get("significant_holm") else "незначајна"
        lines.append(
            f"- {title} (n = {p['n']}): Spearman ρ = {p['spearman_rho']:.3f}, "
            f"p = {p['spearman_p']:.4g}, p_Holm = {holm_txt}, "
            f"95% CI = {ci_txt}. Толкување на ефектот: "
            f"{p['effect_size_interpretation']}. По Holm корекцијата асоцијацијата е {sig}."
        )

    lines.append("")
    lines.append("### Интерпретација")
    lines.append(
        "Доколку по Holm корекцијата се утврди статистички значајна позитивна "
        "корелација меѓу retrieval-метриките и context relevance, тоа укажува дека "
        "автоматските метрики и LLM-судијата се согласуваат во рангирањето на "
        "прашањата според квалитет на retrieval. Послаба или незначајна поврзаност "
        "со correctness би била очекувана доколку добриот retrieval не гарантира "
        "секогаш фактички точен одговор. Практичната значајност се оценува преку "
        "големината на |ρ| и ширината на интервалот на доверба, а не само преку "
        "p-вредноста."
    )
    if report.get("n_judged_max", 0) < 20:
        lines.append("")
        lines.append(
            f"**Ограничување.** Големината на примерокот со judge-оценки "
            f"(n = {report.get('n_judged_max')}) е релативно мала; заклучоците за "
            "значајност треба да се третираат претпазливо. Препорачливо е "
            "проширување кон ≥20–30 (идеално сите 50) прашања."
        )
    return methods + "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Статистичка анализа за RAG евалуација: "
            "Spearman (auto↔judge) и парни тестови (backend↔backend)."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    avj = sub.add_parser(
        "auto-vs-judge",
        help="Примарна анализа: Spearman корелација автоматски ↔ LLM-as-a-Judge",
    )
    avj.add_argument("file", type=Path, help="Резултатска JSON со judge_scores")
    avj.add_argument("--alpha", type=float, default=0.05)
    avj.add_argument("--n-boot", type=int, default=5000, help="Bootstrap репликации за CI")
    avj.add_argument(
        "--report-mk",
        type=Path,
        default=None,
        help="Запиши академски текст на македонски во датотека",
    )
    avj.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Запиши целосен JSON извештај",
    )

    corr = sub.add_parser(
        "correlate",
        help="Еден пар метрики (ист метод како auto-vs-judge)",
    )
    corr.add_argument("file", type=Path)
    corr.add_argument("--retrieval", required=True)
    corr.add_argument("--judge", required=True)
    corr.add_argument("--alpha", type=float, default=0.05)
    corr.add_argument("--n-boot", type=int, default=5000)

    cmp_parser = sub.add_parser(
        "compare",
        help="Секундарно: парна споредба на иста метрика меѓу два backend-а",
    )
    cmp_parser.add_argument("file_a", type=Path)
    cmp_parser.add_argument("file_b", type=Path)
    cmp_parser.add_argument("--metric", required=True)
    cmp_parser.add_argument("--alpha", type=float, default=0.05)

    mk = sub.add_parser(
        "write-methods-mk",
        help="Испиши само методолошкиот дел на македонски (без резултати)",
    )
    mk.add_argument("--out", type=Path, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "write-methods-mk":
        text = macedonian_methods_and_results_section(None)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote methods section to {args.out}")
        return

    if args.command == "compare":
        summary = compare_files(args.file_a, args.file_b, args.metric, alpha=args.alpha)
        print("Comparison result:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "correlate":
        summary = correlate_metrics(
            args.file,
            args.retrieval,
            args.judge,
            alpha=args.alpha,
            n_boot=args.n_boot,
        )
        print("Correlation result:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "auto-vs-judge":
        report = auto_vs_judge_report(args.file, alpha=args.alpha, n_boot=args.n_boot)
        print(format_human_report(report))
        print("\nJSON summary:")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.json_out:
            args.json_out.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"\nWrote JSON report to {args.json_out}")
        if args.report_mk:
            args.report_mk.write_text(
                macedonian_methods_and_results_section(report), encoding="utf-8"
            )
            print(f"Wrote Macedonian section to {args.report_mk}")
        return


if __name__ == "__main__":
    main()
