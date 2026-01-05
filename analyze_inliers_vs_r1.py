import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_pred_file(txt_path: Path) -> Tuple[str, List[str], List[str]]:
    """Return query path, ranked predictions, and positives from a preds txt file."""
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    try:
        idx_query = lines.index("Query path:")
        idx_preds = lines.index("Predictions paths:")
        idx_pos = lines.index("Positives paths:")
    except ValueError as exc:  # pragma: no cover - guard against malformed files
        raise RuntimeError(f"Unexpected layout in {txt_path}") from exc

    query_path = lines[idx_query + 1].strip()

    preds = []
    for line in lines[idx_preds + 1 : idx_pos]:
        line = line.strip()
        if line:
            preds.append(line)

    positives = [line.strip() for line in lines[idx_pos + 1 :] if line.strip()]

    if not preds:
        raise RuntimeError(f"No predictions found in {txt_path}")

    return query_path, preds, positives


def collect_inliers_and_labels(preds_dir: Path, matcher_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load rank-1 inliers and correctness labels for every query in a log run."""
    txt_files = sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not txt_files:
        raise RuntimeError(f"No preds txt files found under {preds_dir}")

    inliers = []
    labels = []

    for txt_file in txt_files:
        _, preds, positives = parse_pred_file(txt_file)
        torch_file = matcher_dir / f"{txt_file.stem}.torch"
        if not torch_file.exists():
            raise FileNotFoundError(f"Missing matcher output: {torch_file}")

        match_results = torch.load(torch_file, map_location="cpu", weights_only=False)
        try:
            num_inliers = float(match_results[0]["num_inliers"])
        except Exception as exc:  # pragma: no cover - guard against malformed torch files
            raise RuntimeError(f"Cannot read num_inliers from {torch_file}") from exc

        inliers.append(num_inliers)
        labels.append(preds[0] in positives)

    return np.array(inliers, dtype=np.float32), np.array(labels, dtype=bool)


def find_best_threshold(inliers: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """Return threshold and accuracy that best separates correct vs wrong queries."""
    candidates = np.percentile(inliers, np.linspace(0, 100, 201))
    best_acc = -1.0
    best_t = candidates[0]
    for t in candidates:
        preds = inliers >= t
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc, best_t = acc, t
    return float(best_t), float(best_acc)


def maybe_compute_auc(inliers: np.ndarray, labels: np.ndarray) -> float:
    """Try computing ROC AUC; return NaN if sklearn is unavailable."""
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels.astype(int), inliers))
    except Exception:
        return float("nan")


def plot_histograms(
    inliers: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    threshold: float,
    title: str,
) -> None:
    correct = inliers[labels]
    wrong = inliers[~labels]

    bins = 30
    plt.figure(figsize=(7, 4))
    plt.hist(correct, bins=bins, alpha=0.6, color="#2a9d8f", label="Correct R@1")
    plt.hist(wrong, bins=bins, alpha=0.6, color="#e76f51", label="Wrong R@1")
    plt.axvline(threshold, color="#264653", linestyle="--", linewidth=1.5, label=f"Thresh: {threshold:.1f}")
    plt.title(title)
    plt.xlabel("# Inliers (rank-1 match)")
    plt.ylabel("Queries")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze inliers vs. retrieval correctness (R@1)")
    parser.add_argument("--log-dir", type=Path, required=True, help="Path to a single run folder under logs/log_dir")
    parser.add_argument(
        "--preds-subdir",
        type=str,
        default="preds",
        help="Subfolder with retrieval txt files (defaults to 'preds')",
    )
    parser.add_argument(
        "--matcher-subdir",
        type=str,
        default="preds_superglue",
        help="Subfolder with .torch matcher outputs (e.g. preds_superglue, preds_loftr)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Where to save the histogram PNG (defaults inside the log dir)"
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help="Optional path to save raw numbers (inliers + labels) as JSON",
    )
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., MixVPR)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--rerank",
        type=str,
        required=True,
        help="Re-ranking / matcher method used (e.g., SuperGlue, LoFTR)",
    )

    args = parser.parse_args()

    preds_dir = args.log_dir / args.preds_subdir
    matcher_dir = args.log_dir / args.matcher_subdir
    inliers, labels = collect_inliers_and_labels(preds_dir, matcher_dir)

    threshold, best_acc = find_best_threshold(inliers, labels)
    auc = maybe_compute_auc(inliers, labels)

    title = f"VPR Method: {args.model} \n Dataset: {args.dataset} \n Re-Ranking Method: {args.rerank}"

    print(f"Total queries: {len(inliers)}")
    print(f"R@1 (retrieval only): {labels.mean() * 100:.2f}%")
    print(f"Mean inliers | correct: {inliers[labels].mean():.1f}")
    print(f"Mean inliers | wrong:   {inliers[~labels].mean():.1f}")
    print(f"Best threshold: {threshold:.1f} -> accuracy {best_acc * 100:.2f}%")
    if not np.isnan(auc):
        print(f"ROC AUC (inliers as score): {auc:.3f}")

    out_png = args.out or (args.log_dir / "inliers_vs_r1.png")
    plot_histograms(inliers, labels, out_png, threshold, title)
    print(f"Saved histogram to {out_png}")

    if args.dump_json:
        payload = {"inliers": inliers.tolist(), "labels": labels.astype(int).tolist(), "threshold": threshold}
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        args.dump_json.write_text(json.dumps(payload, indent=2))
        print(f"Saved raw data to {args.dump_json}")


if __name__ == "__main__":
    main()