import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold


LABELS = ["E", "S", "G", "non_ESG"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TF-IDF + Logistic baseline with per-label threshold tuning for macro-F1")
    parser.add_argument("--train-path", type=str, default="data_set/train.csv")
    parser.add_argument("--test-path", type=str, default="data_set/test.csv")
    parser.add_argument("--sample-submission-path", type=str, default="data_set/sample_submission.csv")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--id-col", type=str, default="id")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-features", type=int, default=120000)
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--apply-non-esg-rule", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_cv(n_splits: int, seed: int):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

        return MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed), "MultilabelStratifiedKFold"
    except Exception:
        return KFold(n_splits=n_splits, shuffle=True, random_state=seed), "KFold (fallback)"


def prevalence_report(train_df: pd.DataFrame) -> Dict[str, float]:
    return train_df[LABELS].mean().to_dict()


def build_model() -> OneVsRestClassifier:
    base = LogisticRegression(
        C=2.0,
        max_iter=2500,
        class_weight="balanced",
        solver="liblinear",
    )
    return OneVsRestClassifier(base)


def find_best_thresholds(y_true: np.ndarray, y_prob: np.ndarray, tmin: float, tmax: float, step: float) -> Dict[str, float]:
    grid = np.arange(tmin, tmax + 1e-12, step)
    best = {}

    for idx, label in enumerate(LABELS):
        label_true = y_true[:, idx]
        label_prob = y_prob[:, idx]

        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            pred = (label_prob >= t).astype(int)
            score = f1_score(label_true, pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = float(t)

        best[label] = round(best_t, 4)

    return best


def apply_thresholds(y_prob: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    preds = np.zeros_like(y_prob, dtype=int)
    for idx, label in enumerate(LABELS):
        preds[:, idx] = (y_prob[:, idx] >= thresholds[label]).astype(int)
    return preds


def maybe_apply_non_esg_rule(preds: np.ndarray) -> np.ndarray:
    fixed = preds.copy()
    esg_any = (fixed[:, 0] + fixed[:, 1] + fixed[:, 2]) > 0
    fixed[esg_any, 3] = 0
    return fixed


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    per_label = [
        f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        for i in range(y_true.shape[1])
    ]
    return float(np.mean(per_label))


def run_cv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    n_splits: int,
    seed: int,
    min_df: int,
    max_features: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]], str]:
    y = train_df[LABELS].values.astype(int)
    oof_prob = np.zeros((len(train_df), len(LABELS)), dtype=float)
    test_prob_folds = []
    fold_scores = []

    cv, cv_name = get_cv(n_splits=n_splits, seed=seed)

    split_iter = cv.split(train_df[text_col], y) if "Multilabel" in cv_name else cv.split(train_df[text_col])

    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        x_tr = train_df.iloc[tr_idx][text_col].fillna("").astype(str).values
        x_va = train_df.iloc[va_idx][text_col].fillna("").astype(str).values
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )

        x_tr_vec = vectorizer.fit_transform(x_tr)
        x_va_vec = vectorizer.transform(x_va)
        x_te_vec = vectorizer.transform(test_df[text_col].fillna("").astype(str).values)

        model = build_model()
        model.fit(x_tr_vec, y_tr)

        va_prob = model.predict_proba(x_va_vec)
        te_prob = model.predict_proba(x_te_vec)

        oof_prob[va_idx] = va_prob
        test_prob_folds.append(te_prob)

        va_pred_default = (va_prob >= 0.5).astype(int)
        fold_macro = macro_f1(y_va, va_pred_default)
        fold_scores.append({"fold": fold, "macro_f1@0.5": round(fold_macro, 6)})

    test_prob = np.mean(np.stack(test_prob_folds, axis=0), axis=0)
    return oof_prob, test_prob, fold_scores, cv_name


def main() -> None:
    args = parse_args()
    ensure_output_dir(args.output_dir)

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample_sub = pd.read_csv(args.sample_submission_path)

    missing = [c for c in LABELS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing label columns in train: {missing}")
    if args.text_col not in train_df.columns or args.text_col not in test_df.columns:
        raise ValueError(f"Text column '{args.text_col}' must exist in both train and test")

    prevalence = prevalence_report(train_df)

    oof_prob, test_prob, fold_scores, cv_name = run_cv(
        train_df=train_df,
        test_df=test_df,
        text_col=args.text_col,
        n_splits=args.n_splits,
        seed=args.seed,
        min_df=args.min_df,
        max_features=args.max_features,
    )

    y_true = train_df[LABELS].values.astype(int)
    thresholds = find_best_thresholds(
        y_true=y_true,
        y_prob=oof_prob,
        tmin=args.threshold_min,
        tmax=args.threshold_max,
        step=args.threshold_step,
    )

    oof_pred = apply_thresholds(oof_prob, thresholds)
    test_pred = apply_thresholds(test_prob, thresholds)

    score_before_rule = macro_f1(y_true, oof_pred)

    non_esg_rule_applied = False
    score_after_rule = score_before_rule
    if args.apply_non_esg_rule:
        oof_pred_rule = maybe_apply_non_esg_rule(oof_pred)
        score_after_rule = macro_f1(y_true, oof_pred_rule)
        if score_after_rule >= score_before_rule:
            oof_pred = oof_pred_rule
            test_pred = maybe_apply_non_esg_rule(test_pred)
            non_esg_rule_applied = True

    submission = sample_sub.copy()
    if args.id_col in sample_sub.columns and args.id_col in test_df.columns:
        submission[args.id_col] = test_df[args.id_col]
    for idx, label in enumerate(LABELS):
        submission[label] = test_pred[:, idx].astype(int)

    oof_prob_df = train_df[[args.id_col]].copy() if args.id_col in train_df.columns else pd.DataFrame(index=np.arange(len(train_df)))
    for idx, label in enumerate(LABELS):
        oof_prob_df[f"{label}_prob"] = oof_prob[:, idx]

    test_prob_df = test_df[[args.id_col]].copy() if args.id_col in test_df.columns else pd.DataFrame(index=np.arange(len(test_df)))
    for idx, label in enumerate(LABELS):
        test_prob_df[f"{label}_prob"] = test_prob[:, idx]

    report = {
        "cv": {
            "strategy": cv_name,
            "n_splits": args.n_splits,
            "seed": args.seed,
            "fold_scores_macro_f1_at_0_5": fold_scores,
        },
        "labels": LABELS,
        "label_prevalence": prevalence,
        "best_thresholds": thresholds,
        "oof_macro_f1_after_threshold_tuning": round(score_before_rule, 6),
        "oof_macro_f1_after_optional_non_esg_rule": round(score_after_rule, 6),
        "non_esg_rule_applied": non_esg_rule_applied,
        "params": {
            "min_df": args.min_df,
            "max_features": args.max_features,
            "threshold_min": args.threshold_min,
            "threshold_max": args.threshold_max,
            "threshold_step": args.threshold_step,
        },
    }

    submission_path = os.path.join(args.output_dir, "submission_tfidf_lr.csv")
    report_path = os.path.join(args.output_dir, "cv_threshold_report.json")
    oof_prob_path = os.path.join(args.output_dir, "oof_probs.csv")
    test_prob_path = os.path.join(args.output_dir, "test_probs.csv")

    submission.to_csv(submission_path, index=False)
    oof_prob_df.to_csv(oof_prob_path, index=False)
    test_prob_df.to_csv(test_prob_path, index=False)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== Label Prevalence ===")
    for label in LABELS:
        print(f"{label}: {prevalence[label]:.4f}")

    print("\n=== Best Thresholds ===")
    for label in LABELS:
        print(f"{label}: {thresholds[label]:.2f}")

    print(f"\nOOF macro-F1 (threshold tuned): {score_before_rule:.6f}")
    print(f"OOF macro-F1 (after optional non-ESG rule): {score_after_rule:.6f}")
    print(f"non-ESG rule applied: {non_esg_rule_applied}")

    print("\nSaved files:")
    print(f"- {submission_path}")
    print(f"- {report_path}")
    print(f"- {oof_prob_path}")
    print(f"- {test_prob_path}")


if __name__ == "__main__":
    main()
