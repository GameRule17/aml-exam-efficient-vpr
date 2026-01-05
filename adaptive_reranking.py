"""
Adaptive Re-Ranking for Visual Place Recognition.

This script implements adaptive re-ranking where:
- If the classifier predicts the query is "difficult" (likely wrong), 
  apply VPR + image matching re-ranking
- If the classifier predicts the query is "easy" (likely correct), 
  use only VPR results (no re-ranking)

This saves computation by only applying expensive image matching 
re-ranking on queries that are likely to benefit from it.
"""

import argparse
import pickle
import json
from pathlib import Path
from typing import List, Tuple, Optional
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def load_classifier(model_dir: Path, model_name: str = "logistic_regressor"):
    """Load the trained classifier, scaler, and optimal threshold."""
    model_path = model_dir / f"{model_name}.pkl"
    scaler_path = model_dir / f"{model_name}_scaler.pkl"
    threshold_path = model_dir / f"{model_name}_threshold.json"
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Load optimal threshold if available
    optimal_threshold = None
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            threshold_data = json.load(f)
            optimal_threshold = threshold_data.get("optimal_threshold")
    
    return model, scaler, optimal_threshold


def parse_pred_file(txt_path: Path) -> Tuple[str, List[str], List[str]]:
    """Return query path, ranked predictions, and positives from a preds txt file."""
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    idx_query = lines.index("Query path:")
    idx_preds = lines.index("Predictions paths:")
    idx_pos = lines.index("Positives paths:")

    query_path = lines[idx_query + 1].strip()
    preds = [l.strip() for l in lines[idx_preds + 1 : idx_pos] if l.strip()]
    positives = [l.strip() for l in lines[idx_pos + 1 :] if l.strip()]

    return query_path, preds, positives


def get_utm_from_path(path: str) -> np.ndarray:
    """Extract UTM coordinates from image path."""
    return np.array([path.split("@")[1], path.split("@")[2]]).astype(np.float32)


def compute_distance(point_A: np.ndarray, point_B: np.ndarray) -> float:
    """Compute Euclidean distance between two points."""
    return float(np.sqrt(((point_A - point_B) ** 2).sum()))


def is_correct_at_k(preds: List[str], positives: List[str], k: int) -> bool:
    """Check if any of the top-k predictions is a positive."""
    return any(pred in positives for pred in preds[:k])


def rerank_by_inliers(
    preds: List[str], 
    match_results: List[dict], 
    num_preds: int
) -> List[str]:
    """Re-rank predictions by number of inliers."""
    # Get inliers for each prediction
    inliers = []
    for i in range(min(num_preds, len(match_results))):
        inliers.append(match_results[i]["num_inliers"])
    
    # Sort by inliers (descending)
    indices = np.argsort(inliers)[::-1]
    
    # Re-order predictions
    reranked = [preds[i] for i in indices]
    
    # Append remaining predictions that weren't re-ranked
    if len(preds) > num_preds:
        reranked.extend(preds[num_preds:])
    
    return reranked


def adaptive_reranking(
    preds_dir: Path,
    matcher_dir: Path,
    classifier_model,
    classifier_scaler,
    num_preds_rerank: int = 100,
    threshold: float = 0.5,
    positive_dist_threshold: int = 25,
    recall_values: List[int] = [1, 5, 10, 20],
) -> dict:
    """
    Perform adaptive re-ranking based on classifier predictions.
    
    Args:
        preds_dir: Directory with VPR prediction txt files
        matcher_dir: Directory with image matching .torch files
        classifier_model: Trained logistic regression model
        classifier_scaler: Fitted scaler for the classifier
        num_preds_rerank: Number of predictions to consider for re-ranking
        threshold: Classifier threshold (below = difficult, apply reranking)
        positive_dist_threshold: Distance threshold for positive match (meters)
        recall_values: K values for Recall@K computation
    
    Returns:
        Dictionary with metrics and statistics
    """
    txt_files = sorted(Path(preds_dir).glob("*.txt"), key=lambda p: int(p.stem))
    total_queries = len(txt_files)
    
    # Metrics for different strategies
    recalls_vpr_only = np.zeros(len(recall_values))
    recalls_rerank_all = np.zeros(len(recall_values))
    recalls_adaptive = np.zeros(len(recall_values))
    
    # Statistics
    num_easy = 0  # Queries predicted as easy (no reranking)
    num_difficult = 0  # Queries predicted as difficult (apply reranking)
    
    # Detailed per-query results
    query_results = []
    
    for txt_file in tqdm(txt_files, desc="Adaptive re-ranking"):
        query_path, preds, positives = parse_pred_file(txt_file)
        
        # Load image matching results
        torch_file = matcher_dir / f"{txt_file.stem}.torch"
        if not torch_file.exists():
            raise FileNotFoundError(f"Missing matcher output: {torch_file}")
        
        match_results = torch.load(torch_file, map_location="cpu", weights_only=False)
        
        # Get number of inliers for rank-1 prediction
        num_inliers = float(match_results[0]["num_inliers"])
        
        # Classifier prediction
        features = np.array([[num_inliers]], dtype=np.float32)
        features_scaled = classifier_scaler.transform(features)
        prob_correct = classifier_model.predict_proba(features_scaled)[0, 1]
        
        # Decide: easy or difficult?
        is_easy = prob_correct >= threshold
        
        if is_easy:
            num_easy += 1
            # Use VPR-only ranking (no re-ranking)
            adaptive_preds = preds
        else:
            num_difficult += 1
            # Apply re-ranking by inliers
            adaptive_preds = rerank_by_inliers(preds, match_results, num_preds_rerank)
        
        # Also compute re-ranked predictions for comparison
        reranked_preds = rerank_by_inliers(preds, match_results, num_preds_rerank)
        
        # Check correctness for each strategy
        for i, k in enumerate(recall_values):
            # VPR only
            if is_correct_at_k(preds, positives, k):
                recalls_vpr_only[i:] += 1
                break
        
        for i, k in enumerate(recall_values):
            # Re-rank all
            if is_correct_at_k(reranked_preds, positives, k):
                recalls_rerank_all[i:] += 1
                break
        
        for i, k in enumerate(recall_values):
            # Adaptive
            if is_correct_at_k(adaptive_preds, positives, k):
                recalls_adaptive[i:] += 1
                break
        
        # Store per-query result
        query_results.append({
            "query_id": int(txt_file.stem),
            "num_inliers": num_inliers,
            "prob_correct": float(prob_correct),
            "is_easy": bool(is_easy),
            "vpr_correct_r1": preds[0] in positives,
            "rerank_correct_r1": reranked_preds[0] in positives,
            "adaptive_correct_r1": adaptive_preds[0] in positives,
        })
    
    # Compute percentages
    recalls_vpr_only = recalls_vpr_only / total_queries * 100
    recalls_rerank_all = recalls_rerank_all / total_queries * 100
    recalls_adaptive = recalls_adaptive / total_queries * 100
    
    results = {
        "total_queries": total_queries,
        "num_easy": num_easy,
        "num_difficult": num_difficult,
        "pct_easy": num_easy / total_queries * 100,
        "pct_difficult": num_difficult / total_queries * 100,
        "threshold": threshold,
        "recall_values": recall_values,
        "recalls_vpr_only": recalls_vpr_only.tolist(),
        "recalls_rerank_all": recalls_rerank_all.tolist(),
        "recalls_adaptive": recalls_adaptive.tolist(),
        "query_results": query_results,
    }
    
    return results


def print_results(results: dict) -> None:
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("ADAPTIVE RE-RANKING RESULTS")
    print("=" * 70)
    
    print(f"\nTotal queries: {results['total_queries']}")
    print(f"Threshold: {results['threshold']}")
    print(f"Easy queries (VPR only): {results['num_easy']} ({results['pct_easy']:.1f}%)")
    print(f"Difficult queries (re-ranked): {results['num_difficult']} ({results['pct_difficult']:.1f}%)")
    
    print("\n" + "-" * 70)
    print("RECALL COMPARISON")
    print("-" * 70)
    
    recall_values = results["recall_values"]
    recalls_vpr = results["recalls_vpr_only"]
    recalls_rerank = results["recalls_rerank_all"]
    recalls_adaptive = results["recalls_adaptive"]
    
    # Header
    header = "Strategy".ljust(20) + "".join([f"R@{k}".rjust(10) for k in recall_values])
    print(header)
    print("-" * len(header))
    
    # VPR only
    row = "VPR only".ljust(20) + "".join([f"{r:.2f}%".rjust(10) for r in recalls_vpr])
    print(row)
    
    # Re-rank all
    row = "Re-rank all".ljust(20) + "".join([f"{r:.2f}%".rjust(10) for r in recalls_rerank])
    print(row)
    
    # Adaptive
    row = "Adaptive".ljust(20) + "".join([f"{r:.2f}%".rjust(10) for r in recalls_adaptive])
    print(row)
    
    print("\n" + "-" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("-" * 70)
    
    # Compare R@1
    vpr_r1 = recalls_vpr[0]
    rerank_r1 = recalls_rerank[0]
    adaptive_r1 = recalls_adaptive[0]
    
    print(f"R@1 VPR only:     {vpr_r1:.2f}%")
    print(f"R@1 Re-rank all:  {rerank_r1:.2f}% (diff: {rerank_r1 - vpr_r1:+.2f}%)")
    print(f"R@1 Adaptive:     {adaptive_r1:.2f}% (diff: {adaptive_r1 - vpr_r1:+.2f}%)")
    
    # Computational savings
    savings = results['pct_easy']
    print(f"\nComputational savings: {savings:.1f}% queries skip re-ranking")
    
    print("=" * 70)


def collect_query_data(
    preds_dir: Path,
    matcher_dir: Path,
    classifier_model,
    classifier_scaler,
    num_preds_rerank: int = 100,
) -> dict:
    """
    Collect all query data needed for threshold sweep analysis.
    
    This pre-computes everything once so we can quickly evaluate different thresholds.
    """
    txt_files = sorted(Path(preds_dir).glob("*.txt"), key=lambda p: int(p.stem))
    total_queries = len(txt_files)
    
    query_data = []
    
    for txt_file in tqdm(txt_files, desc="Collecting query data"):
        query_path, preds, positives = parse_pred_file(txt_file)
        
        # Load image matching results
        torch_file = matcher_dir / f"{txt_file.stem}.torch"
        if not torch_file.exists():
            raise FileNotFoundError(f"Missing matcher output: {torch_file}")
        
        match_results = torch.load(torch_file, map_location="cpu", weights_only=False)
        
        # Get number of inliers for rank-1 prediction
        num_inliers = float(match_results[0]["num_inliers"])
        
        # Classifier prediction (probability that query is correct)
        features = np.array([[num_inliers]], dtype=np.float32)
        features_scaled = classifier_scaler.transform(features)
        prob_correct = classifier_model.predict_proba(features_scaled)[0, 1]
        
        # Pre-compute re-ranked predictions
        reranked_preds = rerank_by_inliers(preds, match_results, num_preds_rerank)
        
        # Store data
        query_data.append({
            "prob_correct": prob_correct,
            "vpr_correct_r1": preds[0] in positives,
            "rerank_correct_r1": reranked_preds[0] in positives,
        })
    
    return {
        "total_queries": total_queries,
        "query_data": query_data,
    }


def threshold_sweep_analysis(
    query_data: dict,
    num_thresholds: int = 101,
) -> dict:
    """
    Analyze R@1 and computational savings across different thresholds.
    
    Args:
        query_data: Pre-computed query data from collect_query_data()
        num_thresholds: Number of threshold values to test
    
    Returns:
        Dictionary with threshold sweep results
    """
    total_queries = query_data["total_queries"]
    queries = query_data["query_data"]
    
    thresholds = np.linspace(0, 1, num_thresholds)
    
    # Pre-compute VPR-only and rerank-all R@1 (these don't depend on threshold)
    vpr_correct = sum(q["vpr_correct_r1"] for q in queries)
    rerank_correct = sum(q["rerank_correct_r1"] for q in queries)
    
    r1_vpr_only = vpr_correct / total_queries * 100
    r1_rerank_all = rerank_correct / total_queries * 100
    
    # Sweep thresholds
    results = []
    
    for t in thresholds:
        # For each query, decide based on threshold
        # If prob_correct >= t -> "easy" -> use VPR result
        # If prob_correct < t -> "difficult" -> use rerank result
        
        adaptive_correct = 0
        num_easy = 0
        
        for q in queries:
            is_easy = q["prob_correct"] >= t
            
            if is_easy:
                num_easy += 1
                if q["vpr_correct_r1"]:
                    adaptive_correct += 1
            else:
                if q["rerank_correct_r1"]:
                    adaptive_correct += 1
        
        r1_adaptive = adaptive_correct / total_queries * 100
        pct_easy = num_easy / total_queries * 100  # Computational savings
        
        results.append({
            "threshold": float(t),
            "r1_adaptive": r1_adaptive,
            "pct_easy": pct_easy,  # % of queries that skip re-ranking
            "pct_reranked": 100 - pct_easy,  # % of queries that are re-ranked
        })
    
    return {
        "thresholds": [r["threshold"] for r in results],
        "r1_adaptive": [r["r1_adaptive"] for r in results],
        "pct_easy": [r["pct_easy"] for r in results],
        "pct_reranked": [r["pct_reranked"] for r in results],
        "r1_vpr_only": r1_vpr_only,
        "r1_rerank_all": r1_rerank_all,
        "total_queries": total_queries,
    }


def find_best_balanced_threshold(
    sweep_results: dict,
    alpha: float = 0.5,
) -> Tuple[int, float]:
    """
    Find the best threshold balancing R@1 improvement and computational savings.
    
    We want to maximize:
        score = alpha * normalized_r1_improvement + (1 - alpha) * normalized_savings
    
    Where:
        - normalized_r1_improvement: How close R@1 adaptive is to R@1 rerank-all (relative to VPR-only)
        - normalized_savings: Percentage of queries skipping re-ranking (0-1)
    
    Args:
        sweep_results: Results from threshold_sweep_analysis()
        alpha: Weight for R@1 (0.5 = balanced, higher = favor R@1, lower = favor savings)
    
    Returns:
        Tuple of (best_index, best_score)
    """
    r1_adaptive = np.array(sweep_results["r1_adaptive"])
    pct_easy = np.array(sweep_results["pct_easy"])
    r1_vpr = sweep_results["r1_vpr_only"]
    r1_rerank = sweep_results["r1_rerank_all"]
    
    # Normalize R@1: 0 = VPR-only level, 1 = Rerank-all level (or better)
    r1_range = r1_rerank - r1_vpr
    if r1_range > 0:
        normalized_r1 = (r1_adaptive - r1_vpr) / r1_range
        normalized_r1 = np.clip(normalized_r1, 0, 1)
    else:
        # If rerank doesn't improve R@1, just use 1 if we match VPR
        normalized_r1 = np.ones_like(r1_adaptive)
    
    # Normalize savings: already in 0-100%, convert to 0-1
    normalized_savings = pct_easy / 100.0
    
    # Compute balanced score
    scores = alpha * normalized_r1 + (1 - alpha) * normalized_savings
    
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    
    return best_idx, best_score


def plot_threshold_analysis(
    sweep_results: dict,
    output_path: Path,
    optimal_threshold: float = None,
    title: str = "Adaptive Re-Ranking: Threshold Analysis",
    alpha: float = 0.5,
) -> Tuple[float, float, float]:
    """
    Plot R@1 and computational savings vs threshold.
    
    Uses a single Y-axis (percentage 0-100) for clarity.
    
    Returns:
        Tuple of (best_threshold, best_r1, best_savings)
    """
    thresholds = sweep_results["thresholds"]
    r1_adaptive = sweep_results["r1_adaptive"]
    pct_easy = sweep_results["pct_easy"]
    r1_vpr = sweep_results["r1_vpr_only"]
    r1_rerank = sweep_results["r1_rerank_all"]
    
    # Find best balanced threshold
    best_idx, best_score = find_best_balanced_threshold(sweep_results, alpha=alpha)
    best_threshold = thresholds[best_idx]
    best_r1 = r1_adaptive[best_idx]
    best_savings = pct_easy[best_idx]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Colors - using well-contrasting colors
    color_r1 = "#2563eb"  # Blue for R@1
    color_savings = "#dc2626"  # Red for savings
    color_vpr = "#6b7280"  # Gray for VPR-only baseline
    color_rerank = "#059669"  # Green for rerank-all baseline
    color_best = "#7c3aed"  # Purple for best threshold
    
    ax.set_xlabel("Threshold (t)", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    
    # Plot R@1 adaptive
    ax.plot(thresholds, r1_adaptive, color=color_r1, linewidth=2.5, label="Adaptive R@1")
    
    # Plot savings (% queries skipping re-ranking)
    ax.plot(thresholds, pct_easy, color=color_savings, linewidth=2.5, linestyle="--", label="Query Savings (skip re-rank)")
    
    # Reference lines for baselines (dashed with longer dashes)
    ax.axhline(y=r1_vpr, color=color_vpr, linestyle="--", linewidth=1.5, dashes=(8, 4), alpha=0.8, label=f"VPR-only R@1 ({r1_vpr:.1f}%)")
    ax.axhline(y=r1_rerank, color=color_rerank, linestyle="--", linewidth=1.5, dashes=(8, 4), alpha=0.8, label=f"Rerank-all R@1 ({r1_rerank:.1f}%)")
    
    # Mark best balanced threshold
    ax.axvline(x=best_threshold, color=color_best, linestyle="-", linewidth=2, alpha=0.7)
    ax.scatter([best_threshold], [best_r1], color=color_best, s=100, zorder=5, marker="o", edgecolors="white", linewidths=1.5)
    ax.scatter([best_threshold], [best_savings], color=color_best, s=100, zorder=5, marker="s", edgecolors="white", linewidths=1.5)
    
    # Annotation for best threshold
    annotation_text = f"Best t={best_threshold:.2f}\nR@1={best_r1:.1f}%\nSavings={best_savings:.1f}%"
    # Position annotation to avoid overlap
    x_offset = 0.05 if best_threshold < 0.7 else -0.15
    ax.annotate(
        annotation_text,
        xy=(best_threshold, best_r1),
        xytext=(best_threshold + x_offset, best_r1 + 5),
        fontsize=10,
        color=color_best,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color_best, alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=color_best, lw=1.5),
    )
    
    # Title and grid
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 105)
    
    # Legend fixed at bottom-left
    ax.legend(loc="lower left", fontsize=10)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved threshold analysis plot to: {output_path}")
    
    return best_threshold, best_r1, best_savings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptive re-ranking for Visual Place Recognition"
    )
    parser.add_argument(
        "--test-log-dir",
        type=Path,
        required=True,
        help="Path to test log directory with preds and matcher outputs",
    )
    parser.add_argument(
        "--classifier-dir",
        type=Path,
        required=True,
        help="Directory containing the trained classifier",
    )
    parser.add_argument(
        "--classifier-name",
        type=str,
        default="logistic_regressor",
        help="Base name of the classifier files",
    )
    parser.add_argument(
        "--preds-subdir",
        type=str,
        default="preds",
        help="Subfolder with retrieval txt files",
    )
    parser.add_argument(
        "--matcher-subdir",
        type=str,
        default="preds_superglue",
        help="Subfolder with .torch matcher outputs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Classifier threshold (queries with prob < threshold are re-ranked). If not provided, uses the optimal threshold from training.",
    )
    parser.add_argument(
        "--num-preds-rerank",
        type=int,
        default=100,
        help="Number of predictions to consider for re-ranking",
    )
    parser.add_argument(
        "--recall-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20],
        help="K values for Recall@K computation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for JSON results (optional)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Output path for threshold sweep plot (enables threshold analysis)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Adaptive Re-Ranking: Threshold Analysis",
        help="Title for the threshold analysis plot",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for R@1 in balanced threshold selection (0-1). 0.5=balanced, higher=favor R@1, lower=favor savings",
    )
    
    args = parser.parse_args()
    
    # Load classifier
    print("Loading classifier...")
    model, scaler, optimal_threshold = load_classifier(args.classifier_dir, args.classifier_name)
    print(f"  Loaded from: {args.classifier_dir}")
    
    # Paths
    preds_dir = args.test_log_dir / args.preds_subdir
    matcher_dir = args.test_log_dir / args.matcher_subdir
    
    print(f"Test data: {args.test_log_dir}")
    
    # If --plot is provided, run threshold sweep analysis
    if args.plot:
        print("\nRunning threshold sweep analysis...")
        
        # Collect query data (this is done once)
        query_data = collect_query_data(
            preds_dir=preds_dir,
            matcher_dir=matcher_dir,
            classifier_model=model,
            classifier_scaler=scaler,
            num_preds_rerank=args.num_preds_rerank,
        )
        
        # Sweep thresholds and compute metrics
        sweep_results = threshold_sweep_analysis(query_data)
        
        # Print summary
        print(f"\nTotal queries: {sweep_results['total_queries']}")
        print(f"R@1 VPR only: {sweep_results['r1_vpr_only']:.2f}%")
        print(f"R@1 Rerank all: {sweep_results['r1_rerank_all']:.2f}%")
        
        if optimal_threshold is not None:
            print(f"\nOptimal threshold from training: {optimal_threshold:.2f}")
        
        # Generate plot and get best balanced threshold
        best_threshold, best_r1, best_savings = plot_threshold_analysis(
            sweep_results=sweep_results,
            output_path=args.plot,
            optimal_threshold=optimal_threshold,
            title=args.title,
            alpha=args.alpha,
        )
        
        print(f"\nBest balanced threshold: {best_threshold:.2f}")
        print(f"  R@1: {best_r1:.2f}%")
        print(f"  Savings: {best_savings:.1f}% queries skip re-ranking")
        
        # Save sweep results as JSON if output provided
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(sweep_results, f, indent=2)
            print(f"Saved sweep results to: {args.output}")
    
    else:
        # Standard mode: run adaptive re-ranking with a single threshold
        
        # Determine threshold to use
        if args.threshold is not None:
            threshold = args.threshold
            print(f"  Using user-specified threshold: {threshold}")
        elif optimal_threshold is not None:
            threshold = optimal_threshold
            print(f"  Using optimal threshold from training: {threshold}")
        else:
            threshold = 0.5
            print(f"  Using default threshold: {threshold}")
        
        # Run adaptive re-ranking
        results = adaptive_reranking(
            preds_dir=preds_dir,
            matcher_dir=matcher_dir,
            classifier_model=model,
            classifier_scaler=scaler,
            num_preds_rerank=args.num_preds_rerank,
            threshold=threshold,
            recall_values=args.recall_values,
        )
        
        # Print results
        print_results(results)
        
        # Save results if output path provided
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            # Remove per-query results for smaller file (optional)
            results_to_save = {k: v for k, v in results.items() if k != "query_results"}
            results_to_save["query_results_summary"] = {
                "total": len(results["query_results"]),
                "saved_to": "full results in memory only"
            }
            with open(args.output, "w") as f:
                json.dump(results_to_save, f, indent=2)
            print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()
