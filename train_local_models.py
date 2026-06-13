from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


DATA_DIR = Path("data")
RESOURCE_DIR = Path("resources")
RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CHUNK_SIZE = 10000


def debug_print(message: str) -> None:
    print(f"[debug] {message}", flush=True)


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("Feature_")]


def split_moons(frame: pd.DataFrame, validation_size: float = 0.1, gap: int = 4) -> tuple[int, int]:
    moons = sorted(frame["moon"].unique())
    if len(moons) <= gap + 2:
        raise ValueError("Not enough moons to create a validation split with the requested gap.")

    validation_count = max(1, int(len(moons) * validation_size))
    validation_start_index = max(gap + 1, len(moons) - validation_count)
    validation_start_moon = int(moons[validation_start_index])
    train_end_moon = int(moons[validation_start_index - gap - 1])

    if train_end_moon >= validation_start_moon:
        raise ValueError("Invalid moon split: training period overlaps validation period.")

    return train_end_moon, validation_start_moon


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    x_train = pd.read_parquet(DATA_DIR / "X.reduced.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y.reduced.parquet")
    return x_train, y_train


def add_basic_row_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Preprocessing: add a few cheap row-wise statistics to the raw feature table.

    The computation is chunked to avoid building a huge temporary float64 array for the
    full dataset at once.
    """
    feature_columns = get_feature_columns(frame)
    missing_count_parts: list[np.ndarray] = []
    feature_mean_parts: list[np.ndarray] = []

    for start_index in range(0, len(frame), FEATURE_CHUNK_SIZE):
        end_index = min(start_index + FEATURE_CHUNK_SIZE, len(frame))
        chunk = frame.iloc[start_index:end_index][feature_columns].to_numpy(dtype=np.float32, copy=False)
        missing_count_parts.append(np.isnan(chunk).sum(axis=1).astype(np.int32, copy=False))
        feature_mean_parts.append(np.nanmean(chunk, axis=1).astype(np.float32, copy=False))

    features = frame
    features["feature_missing_count"] = np.concatenate(missing_count_parts)
    features["feature_missing_rate"] = features["feature_missing_count"] / max(1, len(feature_columns))
    features["feature_mean"] = np.concatenate(feature_mean_parts)

    return features


def model_artifact_path(backend: str) -> Path:
    return RESOURCE_DIR / f"model_{backend}.joblib"


def metrics_artifact_path(backend: str) -> Path:
    return RESOURCE_DIR / f"metrics_{backend}.json"


def save_model_artifacts(model: Any, backend: str) -> Path:
    named_path = model_artifact_path(backend)
    alias_path = RESOURCE_DIR / "model.joblib"
    joblib.dump(model, named_path)
    joblib.dump(model, alias_path)
    return named_path


def save_metrics_artifacts(metrics: dict[str, Any], backend: str) -> Path:
    named_path = metrics_artifact_path(backend)
    alias_path = RESOURCE_DIR / "metrics.json"
    payload = json.dumps(metrics, indent=2, ensure_ascii=False)
    named_path.write_text(payload, encoding="utf-8")
    alias_path.write_text(payload, encoding="utf-8")
    return named_path


def build_ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(add_basic_row_features, validate=False)),
            ("imputer", SimpleImputer(strategy="median")),
            ("variance_threshold", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def train_ridge(alpha: float, validation_size: float, gap: int) -> dict[str, Any]:
    x_train, y_train = load_data()

    debug_print(f"selected backend=ridge")
    debug_print(f"loaded X_train shape={x_train.shape}, y_train shape={y_train.shape}")

    # Preprocessing: raw anonymous features are expanded with a few cheap row statistics.
    feature_columns = get_feature_columns(x_train)
    debug_print(f"feature_count={len(feature_columns)}")

    train_end_moon, validation_start_moon = split_moons(x_train, validation_size=validation_size, gap=gap)
    moon_values = x_train["moon"].to_numpy()
    train_mask = moon_values <= train_end_moon
    validation_mask = moon_values >= validation_start_moon

    x_train_frame = x_train.loc[train_mask, feature_columns]
    y_train_frame = y_train.loc[train_mask, "target"]
    x_validation_frame = x_train.loc[validation_mask, feature_columns]
    y_validation_frame = y_train.loc[validation_mask, "target"]

    debug_print(
        "split completed: "
        f"train_rows={int(train_mask.sum())}, validation_rows={int(validation_mask.sum())}, "
        f"train_moon_min={int(moon_values[train_mask].min())}, train_moon_max={int(moon_values[train_mask].max())}, "
        f"validation_moon_min={int(moon_values[validation_mask].min())}, validation_moon_max={int(moon_values[validation_mask].max())}"
    )

    # Training: ridge is still a fast baseline, now fed with the extra engineered columns.
    pipeline = build_ridge_pipeline(alpha=alpha)
    pipeline.fit(x_train_frame, y_train_frame)

    validation_prediction = pd.Series(pipeline.predict(x_validation_frame)).clip(-1, 1)
    validation_score = validation_prediction.corr(pd.Series(y_validation_frame).reset_index(drop=True))
    validation_rmse = mean_squared_error(y_validation_frame, validation_prediction, squared=False)

    model_path = save_model_artifacts(pipeline, "ridge")

    metrics = {
        "backend": "ridge",
        "alpha": alpha,
        "validation_size": validation_size,
        "gap": gap,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "validation_score": None if pd.isna(validation_score) else float(validation_score),
        "validation_rmse": float(validation_rmse),
        "model_path": str(model_path),
    }
    save_metrics_artifacts(metrics, "ridge")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def train_lightgbm(
    validation_size: float,
    gap: int,
    use_cuda: bool,
    num_boost_round: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    feature_fraction: float,
    bagging_fraction: float,
    bagging_freq: int,
    min_child_samples: int,
    lambda_l1: float,
    lambda_l2: float,
    random_state: int,
) -> dict[str, Any]:
    if lgb is None:
        raise ImportError("lightgbm is not installed. Install it with `pip install lightgbm`.")

    x_train, y_train = load_data()

    debug_print(f"selected backend=lightgbm")
    debug_print(f"loaded X_train shape={x_train.shape}, y_train shape={y_train.shape}")

    # Preprocessing: same lightweight row statistics as the ridge pipeline.
    feature_columns = get_feature_columns(x_train)
    debug_print(f"feature_count={len(feature_columns)}")

    train_end_moon, validation_start_moon = split_moons(x_train, validation_size=validation_size, gap=gap)
    moon_values = x_train["moon"].to_numpy()
    train_mask = moon_values <= train_end_moon
    validation_mask = moon_values >= validation_start_moon

    x_train_frame = x_train.loc[train_mask, feature_columns].astype(np.float32, copy=False)
    y_train_frame = y_train.loc[train_mask, "target"].astype(np.float32, copy=False)
    x_validation_frame = x_train.loc[validation_mask, feature_columns].astype(np.float32, copy=False)
    y_validation_frame = y_train.loc[validation_mask, "target"].astype(np.float32, copy=False)

    debug_print(
        "split completed: "
        f"train_rows={int(train_mask.sum())}, validation_rows={int(validation_mask.sum())}, "
        f"train_moon_min={int(moon_values[train_mask].min())}, train_moon_max={int(moon_values[train_mask].max())}, "
        f"validation_moon_min={int(moon_values[validation_mask].min())}, validation_moon_max={int(moon_values[validation_mask].max())}"
    )

    # Training: LightGBM can consume NaNs directly, so we keep this branch memory-light.
    if use_cuda:
        debug_print("CUDA flag is ignored in this CPU LightGBM pipeline; training on CPU.")

    model = lgb.LGBMRegressor(
        objective="regression",
        learning_rate=learning_rate,
        n_estimators=num_boost_round,
        num_leaves=num_leaves,
        max_depth=max_depth,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        bagging_freq=bagging_freq,
        min_child_samples=min_child_samples,
        reg_alpha=lambda_l1,
        reg_lambda=lambda_l2,
        random_state=random_state,
        verbosity=-1,
    )

    model.fit(x_train_frame, y_train_frame, eval_set=[(x_validation_frame, y_validation_frame)])

    validation_prediction = pd.Series(model.predict(x_validation_frame)).clip(-1, 1)
    validation_score = validation_prediction.corr(pd.Series(y_validation_frame).reset_index(drop=True))
    validation_rmse = mean_squared_error(y_validation_frame, validation_prediction, squared=False)

    model_path = save_model_artifacts(model, "lightgbm")

    metrics = {
        "backend": "lightgbm",
        "use_cuda": use_cuda,
        "num_boost_round": num_boost_round,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "min_child_samples": min_child_samples,
        "lambda_l1": lambda_l1,
        "lambda_l2": lambda_l2,
        "validation_size": validation_size,
        "gap": gap,
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "validation_score": None if pd.isna(validation_score) else float(validation_score),
        "validation_rmse": float(validation_rmse),
        "model_path": str(model_path),
    }
    save_metrics_artifacts(metrics, "lightgbm")

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return metrics


def train(
    backend: str = "lightgbm",
    alpha: float = 1.0,
    validation_size: float = 0.1,
    gap: int = 4,
    use_cuda: bool = True,
    num_boost_round: int = 300,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
    max_depth: int = -1,
    feature_fraction: float = 0.8,
    bagging_fraction: float = 0.8,
    bagging_freq: int = 1,
    min_child_samples: int = 20,
    lambda_l1: float = 0.0,
    lambda_l2: float = 0.0,
    random_state: int = 42,
) -> dict[str, Any]:
    backend = backend.lower().strip()
    if backend == "ridge":
        return train_ridge(alpha=alpha, validation_size=validation_size, gap=gap)
    if backend == "lightgbm":
        return train_lightgbm(
            validation_size=validation_size,
            gap=gap,
            use_cuda=use_cuda,
            num_boost_round=num_boost_round,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            min_child_samples=min_child_samples,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
            random_state=random_state,
        )

    raise ValueError("backend must be either 'ridge' or 'lightgbm'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a local ridge or lightGBM baseline and save models under resources/.")
    parser.add_argument("--backend", choices=["ridge", "lightgbm"], default="lightgbm", help="Training backend.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength.")
    parser.add_argument("--validation-size", type=float, default=0.1, help="Fraction of moons used for validation.")
    parser.add_argument("--gap", type=int, default=4, help="Moon gap between train and validation.")
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, default=True, help="Use CUDA for lightGBM if available.")
    parser.add_argument("--num-boost-round", type=int, default=300, help="Number of boosting rounds for lightGBM.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate for lightGBM.")
    parser.add_argument("--num-leaves", type=int, default=63, help="Number of leaves for lightGBM.")
    parser.add_argument("--max-depth", type=int, default=-1, help="Maximum tree depth for lightGBM.")
    parser.add_argument("--feature-fraction", type=float, default=0.8, help="Feature subsampling ratio for lightGBM.")
    parser.add_argument("--bagging-fraction", type=float, default=0.8, help="Row subsampling ratio for lightGBM.")
    parser.add_argument("--bagging-freq", type=int, default=1, help="Bagging frequency for lightGBM.")
    parser.add_argument("--min-child-samples", type=int, default=20, help="Minimum child samples for lightGBM.")
    parser.add_argument("--lambda-l1", type=float, default=0.0, help="L1 regularization for lightGBM.")
    parser.add_argument("--lambda-l2", type=float, default=0.0, help="L2 regularization for lightGBM.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        backend=args.backend,
        alpha=args.alpha,
        validation_size=args.validation_size,
        gap=args.gap,
        use_cuda=args.use_cuda,
        num_boost_round=args.num_boost_round,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        min_child_samples=args.min_child_samples,
        lambda_l1=args.lambda_l1,
        lambda_l2=args.lambda_l2,
        random_state=args.random_state,
    )
