"""
Train a logistic regressor to predict query correctness (R@1) based on
the number of inliers from image matching.

Extension 6.1: Use the training set of svox to train a classifier that
can distinguish between correct and incorrect retrievals.
"""

import argparse
import itertools
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def parse_pred_file(txt_path: Path) -> Tuple[str, List[str], List[str]]:
    """Return query path, ranked predictions, and positives from a preds txt file."""
    lines = txt_path.read_text(encoding="utf-8").splitlines()
    try:
        idx_query = lines.index("Query path:")
        idx_preds = lines.index("Predictions paths:")
        idx_pos = lines.index("Positives paths:")
    except ValueError as exc:
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


def collect_features_and_labels(
    preds_dir: Path, matcher_dir: Path, num_preds_rerank: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect features (num_inliers) and labels for all queries.
    
    Returns:
        features: (N, 1) array of inlier counts for rank-1 prediction
        vpr_labels: (N,) boolean array, True if VPR-only R@1 is correct
        rerank_labels: (N,) boolean array, True if R@1 after re-ranking is correct
    """
    txt_files = sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not txt_files:
        raise RuntimeError(f"No preds txt files found under {preds_dir}")

    inliers = []
    vpr_labels = []
    rerank_labels = []

    for txt_file in txt_files:
        _, preds, positives = parse_pred_file(txt_file)
        torch_file = matcher_dir / f"{txt_file.stem}.torch"
        if not torch_file.exists():
            raise FileNotFoundError(f"Missing matcher output: {torch_file}")

        match_results = torch.load(torch_file, map_location="cpu", weights_only=False)
        try:
            num_inliers = float(match_results[0]["num_inliers"])
        except Exception as exc:
            raise RuntimeError(f"Cannot read num_inliers from {torch_file}") from exc

        inliers.append(num_inliers)
        
        # VPR-only label
        vpr_labels.append(preds[0] in positives)
        
        # Re-ranking label: re-rank by inliers and check rank-1
        inliers_list = []
        for i in range(min(num_preds_rerank, len(match_results), len(preds))):
            inliers_list.append(match_results[i]["num_inliers"])
        reranked_indices = np.argsort(inliers_list)[::-1]
        reranked_preds = [preds[i] for i in reranked_indices]
        rerank_labels.append(reranked_preds[0] in positives)

    # Features as 2D array for sklearn
    features = np.array(inliers, dtype=np.float32).reshape(-1, 1)
    vpr_labels = np.array(vpr_labels, dtype=bool)
    rerank_labels = np.array(rerank_labels, dtype=bool)

    return features, vpr_labels, rerank_labels


def train_logistic_regressor(
    features: np.ndarray,
    labels: np.ndarray,
    val_features: np.ndarray = None,
    val_labels: np.ndarray = None,
    tune_hyperparams: bool = False,
    random_state: int = 42,
) -> Tuple[LogisticRegression, StandardScaler, dict]:
    """
    Train a logistic regressor with feature standardization.
    
    Args:
        features: (N, D) feature array for training
        labels: (N,) boolean labels for training
        val_features: (M, D) feature array for validation (optional)
        val_labels: (M,) boolean labels for validation (optional)
        tune_hyperparams: whether to perform hyperparameter tuning
        random_state: seed for reproducibility
    
    Returns:
        Trained LogisticRegression model, fitted StandardScaler, and best hyperparams dict
    """
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Default hyperparameters
    best_params = {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}

    if tune_hyperparams and val_features is not None and val_labels is not None:
        # Scale validation features with the same scaler
        val_features_scaled = scaler.transform(val_features)
        
        # Hyperparameter ranges
        C_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        penalty_values = ["l1", "l2", "elasticnet", None]
        solver_values = ["lbfgs", "liblinear", "saga"]
        
        # Compatibility rules: solver -> supported penalties
        solver_penalty_compat = {
            "lbfgs": ["l2", None],
            "liblinear": ["l1", "l2"],
            "saga": ["l1", "l2", "elasticnet", None],
        }
        
        best_val_auc = -1.0
        
        print("  Tuning hyperparameters on validation set...")
        
        for solver in solver_values:
            for penalty in penalty_values:
                # Skip incompatible solver-penalty combinations
                if penalty not in solver_penalty_compat[solver]:
                    continue
                
                for C in C_values:
                    # Build params dict
                    params = {"solver": solver, "penalty": penalty, "C": C}
                    
                    # elasticnet requires l1_ratio
                    if penalty == "elasticnet":
                        params["l1_ratio"] = 0.5
                    
                    try:
                        model_temp = LogisticRegression(
                            C=C,
                            penalty=penalty,
                            solver=solver,
                            l1_ratio=params.get("l1_ratio", None),
                            random_state=random_state,
                            max_iter=1000,
                            class_weight="balanced", # Higher error cost for minority class
                        )
                        model_temp.fit(features_scaled, labels.astype(int))
                        val_probs = model_temp.predict_proba(val_features_scaled)[:, 1]
                        val_auc = roc_auc_score(val_labels.astype(int), val_probs)
                    except Exception:
                        # Skip convergence issues or other errors
                        continue
                    
                    param_str = f"solver={solver}, penalty={penalty}, C={C}"
                    if penalty == "elasticnet":
                        param_str += f", l1_ratio={params['l1_ratio']}"
                    print(f"    {param_str}: ROC AUC = {val_auc:.4f}")
                    
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        best_params = params.copy()
        
        print(f"  Best params: {best_params} (validation ROC AUC: {best_val_auc:.4f})")

    # Train final model with best hyperparameters
    model = LogisticRegression(
        C=best_params["C"],
        penalty=best_params["penalty"],
        solver=best_params["solver"],
        l1_ratio=best_params.get("l1_ratio", None),
        random_state=random_state,
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(features_scaled, labels.astype(int))

    return model, scaler, best_params


def evaluate_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Evaluate the trained model and return metrics."""
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)[:, 1]

    labels_int = labels.astype(int)
    accuracy = accuracy_score(labels_int, predictions)
    
    # ROC AUC (handle edge case where only one class is present)
    try:
        auc = roc_auc_score(labels_int, probabilities)
    except ValueError:
        auc = float("nan")

    report = classification_report(labels_int, predictions, output_dict=True)

    return {
        "accuracy": accuracy,
        "roc_auc": auc,
        "classification_report": report,
        "n_samples": len(labels),
        "n_correct": int(labels.sum()),
        "n_wrong": int((~labels).sum()),
    }


def find_optimal_threshold(
    model: LogisticRegression,
    scaler: StandardScaler,
    features: np.ndarray,
    labels: np.ndarray,
    rerank_labels: np.ndarray,
    alpha: float = 0.5,
    num_thresholds: int = 100,
) -> Tuple[float, dict]:
    """
    Find the optimal classification threshold on the validation set.
    
    The threshold is optimized to balance R@1 improvement and computational savings:
        score = alpha * normalized_R@1_improvement + (1 - alpha) * normalized_savings
    
    The classifier outputs P(correct | x) = sigmoid(w * x + b).
    We need to find t* such that:
    - If P(correct | x) >= t* -> predict "easy" (use VPR result, skip re-ranking)
    - If P(correct | x) < t* -> predict "difficult" (apply re-ranking)
    
    Args:
        model: Trained logistic regression model
        scaler: Fitted scaler
        features: Validation features (num_inliers)
        labels: Validation labels for VPR-only (True = correct R@1 with VPR)
        rerank_labels: Validation labels after re-ranking (True = correct R@1 after rerank)
        alpha: Weight for R@1 (0.5 = balanced, higher = favor R@1, lower = favor savings)
        num_thresholds: Number of threshold values to test
    
    Returns:
        Optimal threshold and dict with all threshold results
    """
    features_scaled = scaler.transform(features)
    probabilities = model.predict_proba(features_scaled)[:, 1]
    
    total_queries = len(labels)
    
    # Baseline R@1 values
    r1_vpr_only = labels.sum() / total_queries * 100
    r1_rerank_all = rerank_labels.sum() / total_queries * 100
    
    # Test thresholds from 0 to 1
    thresholds = np.linspace(0, 1, num_thresholds + 1)
    results = []
    
    for t in thresholds:
        # For each query:
        # - If prob >= t -> "easy" -> use VPR result
        # - If prob < t -> "difficult" -> use rerank result
        is_easy = probabilities >= t
        
        # Adaptive correctness: use VPR label if easy, rerank label if difficult
        adaptive_correct = np.where(is_easy, labels, rerank_labels)
        
        r1_adaptive = adaptive_correct.sum() / total_queries * 100
        pct_easy = is_easy.sum() / total_queries * 100  # % queries skipping re-ranking
        
        # Compute balanced score
        r1_range = r1_rerank_all - r1_vpr_only
        if r1_range > 0:
            normalized_r1 = (r1_adaptive - r1_vpr_only) / r1_range
            normalized_r1 = np.clip(normalized_r1, 0, 1)
        else:
            # If rerank doesn't improve R@1, use 1 if we match or beat VPR
            normalized_r1 = 1.0 if r1_adaptive >= r1_vpr_only else 0.0
        
        normalized_savings = pct_easy / 100.0
        
        score = alpha * normalized_r1 + (1 - alpha) * normalized_savings
        
        results.append({
            "threshold": float(t),
            "r1_adaptive": float(r1_adaptive),
            "pct_easy": float(pct_easy),
            "normalized_r1": float(normalized_r1),
            "normalized_savings": float(normalized_savings),
            "score": float(score),
        })
    
    # Find optimal threshold (maximize balanced score)
    best_result = max(results, key=lambda x: x["score"])
    optimal_threshold = best_result["threshold"]
    
    return optimal_threshold, {
        "optimal_threshold": optimal_threshold,
        "alpha": alpha,
        "r1_vpr_only": r1_vpr_only,
        "r1_rerank_all": r1_rerank_all,
        "best_metrics": best_result,
        "all_results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a logistic regressor to predict R@1 correctness from inliers"
    )
    parser.add_argument(
        "--train-log-dir",
        type=Path,
        required=True,
        help="Path to a training run folder (e.g., logs/log_dir/2025-xx-xx_xx-xx-xx)",
    )
    parser.add_argument(
        "--val-log-dir",
        type=Path,
        default=None,
        help="Path to a validation run folder for hyperparameter tuning (optional)",
    )
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
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the trained model (defaults to train-log-dir)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="logistic_regressor",
        help="Base name for saved model files",
    )
    parser.add_argument(
        "--tune-hyperparams",
        action="store_true",
        help="Tune hyperparameters (C, penalty, solver) using validation set",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight for R@1 in threshold optimization (0-1). 0.5=balanced, higher=favor R@1, lower=favor savings",
    )

    args = parser.parse_args()

    # Paths
    preds_dir = args.train_log_dir / args.preds_subdir
    matcher_dir = args.train_log_dir / args.matcher_subdir
    output_dir = args.output_dir or args.train_log_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Logistic Regressor for R@1 Correctness Prediction")
    print("=" * 60)
    print(f"Training data: {args.train_log_dir}")
    if args.val_log_dir:
        print(f"Validation data: {args.val_log_dir}")
    print(f"Preds subdir: {args.preds_subdir}")
    print(f"Matcher subdir: {args.matcher_subdir}")
    print(f"Hyperparameter tuning: {args.tune_hyperparams}")
    print(f"Alpha (R@1 vs savings): {args.alpha}")
    print()

    # Collect training data
    print("Collecting features and labels from training set...")
    features, vpr_labels, rerank_labels = collect_features_and_labels(preds_dir, matcher_dir)
    print(f"  Total samples: {len(vpr_labels)}")
    print(f"  VPR-only R@1 correct: {vpr_labels.sum()} ({vpr_labels.mean() * 100:.2f}%)")
    print(f"  Re-ranked R@1 correct: {rerank_labels.sum()} ({rerank_labels.mean() * 100:.2f}%)")
    print()

    # Collect validation data if provided
    val_features, val_vpr_labels, val_rerank_labels = None, None, None
    if args.val_log_dir:
        val_preds_dir = args.val_log_dir / args.preds_subdir
        val_matcher_dir = args.val_log_dir / args.matcher_subdir
        print("Collecting features and labels from validation set...")
        val_features, val_vpr_labels, val_rerank_labels = collect_features_and_labels(val_preds_dir, val_matcher_dir)
        print(f"  Total samples: {len(val_vpr_labels)}")
        print(f"  VPR-only R@1 correct: {val_vpr_labels.sum()} ({val_vpr_labels.mean() * 100:.2f}%)")
        print(f"  Re-ranked R@1 correct: {val_rerank_labels.sum()} ({val_rerank_labels.mean() * 100:.2f}%)")
        print()

    # Train the model (using VPR labels as target - predicting if VPR is correct)
    print("Training logistic regressor...")
    model, scaler, best_params = train_logistic_regressor(
        features,
        vpr_labels,
        val_features=val_features,
        val_labels=val_vpr_labels,
        tune_hyperparams=args.tune_hyperparams,
    )
    print("  Training complete!")
    print()

    # Evaluate on training set (to check fit)
    print("Evaluating on training set...")
    train_metrics = evaluate_model(model, scaler, features, vpr_labels)
    print(f"  Accuracy: {train_metrics['accuracy'] * 100:.2f}%")
    if not np.isnan(train_metrics["roc_auc"]):
        print(f"  ROC AUC: {train_metrics['roc_auc']:.3f}")
    print()

    # Evaluate on validation set if provided
    val_metrics = None
    optimal_threshold = 0.5  # Default threshold
    threshold_results = None
    
    if val_features is not None and val_vpr_labels is not None:
        print("Evaluating on validation set...")
        val_metrics = evaluate_model(model, scaler, val_features, val_vpr_labels)
        print(f"  Accuracy: {val_metrics['accuracy'] * 100:.2f}%")
        if not np.isnan(val_metrics["roc_auc"]):
            print(f"  ROC AUC: {val_metrics['roc_auc']:.3f}")
        print()
        
        # Find optimal threshold on validation set (balancing R@1 and savings)
        print(f"Finding optimal threshold (alpha={args.alpha})...")
        optimal_threshold, threshold_results = find_optimal_threshold(
            model, scaler, val_features, val_vpr_labels, val_rerank_labels, alpha=args.alpha
        )
        best_metrics = threshold_results["best_metrics"]
        print(f"  R@1 VPR-only: {threshold_results['r1_vpr_only']:.2f}%")
        print(f"  R@1 Rerank-all: {threshold_results['r1_rerank_all']:.2f}%")
        print(f"  Optimal threshold: {optimal_threshold:.3f}")
        print(f"  At this threshold:")
        print(f"    R@1 Adaptive: {best_metrics['r1_adaptive']:.2f}%")
        print(f"    Query Savings: {best_metrics['pct_easy']:.1f}% skip re-ranking")
        print(f"    Balanced Score: {best_metrics['score']:.3f}")
        print()

    # Model coefficients
    print("Model parameters:")
    print(f"  Solver: {best_params['solver']}")
    print(f"  Penalty: {best_params['penalty']}")
    print(f"  Regularization C: {best_params['C']}")
    if best_params.get('l1_ratio') is not None:
        print(f"  L1 ratio: {best_params['l1_ratio']}")
    print(f"  Coefficient (inliers): {model.coef_[0][0]:.4f}")
    print(f"  Intercept: {model.intercept_[0]:.4f}")
    print(f"  Scaler mean: {scaler.mean_[0]:.4f}")
    print(f"  Scaler std: {scaler.scale_[0]:.4f}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print()

    # Save the model and scaler
    model_path = output_dir / f"{args.model_name}.pkl"
    scaler_path = output_dir / f"{args.model_name}_scaler.pkl"
    metrics_path = output_dir / f"{args.model_name}_metrics.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to: {model_path}")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to: {scaler_path}")

    all_metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "best_hyperparams": best_params,
        "optimal_threshold": optimal_threshold,
        "threshold_optimization": {
            "alpha": args.alpha if val_features is not None else None,
            "best_metrics": threshold_results["best_metrics"] if threshold_results else None,
            "r1_vpr_only": threshold_results["r1_vpr_only"] if threshold_results else None,
            "r1_rerank_all": threshold_results["r1_rerank_all"] if threshold_results else None,
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")
    
    # Save optimal threshold separately for easy loading
    threshold_path = output_dir / f"{args.model_name}_threshold.json"
    threshold_data = {
        "optimal_threshold": optimal_threshold,
        "alpha": args.alpha if val_features is not None else None,
    }
    with open(threshold_path, "w") as f:
        json.dump(threshold_data, f, indent=2)
    print(f"Saved optimal threshold to: {threshold_path}")

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
