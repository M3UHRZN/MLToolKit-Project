
# This file contains the "core" machine learning logic.
# No GUI code here: it focuses on preprocessing, training, and evaluation.
# The GUI (app.py) calls train_evaluate() and then displays its outputs.
# ------------------------------

from __future__ import annotations  # Allows type hints like -> TrainConfig before the class is defined

from dataclasses import dataclass, field  # Dataclasses for clean config objects
from typing import Dict, List, Tuple, Optional  # Type hints (clarity + readability)

import numpy as np  # Numerical utilities (unique counts, concatenation, etc.)
import pandas as pd  # DataFrame operations (dtypes, filtering, binning)

# Train/test split helper
from sklearn.model_selection import train_test_split

# ColumnTransformer lets us apply different preprocessing to numeric vs categorical columns
from sklearn.compose import ColumnTransformer

# Pipeline chains preprocessing + model into a single object
from sklearn.pipeline import Pipeline

# Preprocessing tools:
# - OneHotEncoder for categorical features
# - StandardScaler / MinMaxScaler for numeric scaling
# - LabelEncoder to turn string labels into integer classes
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, LabelEncoder

# SimpleImputer fills missing values (median for numeric, most_frequent for categorical)
from sklearn.impute import SimpleImputer

# Classification models required by the project
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Evaluation metrics required by the project
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


@dataclass
class MLPParams:
    hidden_layers: int = 2   # 1..4
    n1: int = 32
    n2: int = 16
    n3: int = 8
    n4: int = 4
    activation: str = "relu"  # relu/tanh/logistic
    learning_rate_init: float = 0.001
    max_iter: int = 500


@dataclass
class TrainConfig:
    """
    A single config object holding everything needed for training:
    - which column is the target
    - train/test split
    - preprocessing choices
    - which models to run
    - MLP settings
    - optional binning for numeric target columns
    """
    target_col: str
    test_size: float = 0.2
    random_state: int = 42

    # Preprocessing options
    use_onehot: bool = True
    use_norm: bool = True
    scaler_type: str = "standard"  # standard | minmax

    # Model selection
    use_perceptron: bool = True
    use_mlp: bool = True
    use_tree: bool = True

    # Numeric target binning (to convert numeric labels into classes)
    bin_numeric_target: bool = False
    bin_count: int = 3             # allowed: 3, 5, 7
    bin_strategy: str = "quantile" # quantile (equal-frequency) | uniform (equal-width)

    # Use default_factory to avoid "mutable default" dataclass issues
    mlp: MLPParams = field(default_factory=MLPParams)


def recommend_target_column(df: pd.DataFrame) -> str:
    """
    Simple heuristic to guess a label column:
    - prefer columns with a relatively small number of unique values
    - if uncertain, fallback to the last column
    """
    n = len(df)
    candidates = []
    for c in df.columns:
        uniq = df[c].dropna().nunique()
        # Likely a label if the number of unique values is not too large
        if 1 < uniq <= min(30, max(5, int(0.2 * n))):
            candidates.append((uniq, c))

    if candidates:
        candidates.sort(key=lambda x: x[0])  # pick smallest unique count
        last = df.columns[-1]
        # If last column is also a candidate, prefer it (common dataset convention)
        for _, c in candidates:
            if c == last:
                return c
        return candidates[0][1]

    return df.columns[-1]


def dataset_summary(df: pd.DataFrame, file_path: Optional[str] = None) -> str:
    """Create a readable summary string for the GUI (shape, missing, column types)."""
    miss_total = int(df.isna().sum().sum())
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    lines = []
    if file_path:
        lines.append(f"File: {file_path}")
    lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} cols")
    lines.append(f"Missing values (total): {miss_total}")
    lines.append("")
    lines.append(f"Numeric columns ({len(num_cols)}): {num_cols}")
    lines.append(f"Categorical/Bool columns ({len(cat_cols)}): {cat_cols}")
    lines.append("")
    lines.append("Tip: Target should be the label column you want to predict.")
    return "\n".join(lines)


def class_distribution_text(df: pd.DataFrame, target_col: str, top_k: int = 20) -> str:
    """Show the first N class counts for the selected target column."""
    if target_col not in df.columns:
        return "Select a valid target to see distribution."
    y = df[target_col]
    vc = y.astype(str).value_counts(dropna=False).head(top_k)
    return "Top class counts (first 20):\n\n" + vc.to_string()


def _safe_onehot():
    """
    Small compatibility helper:
    - Newer sklearn uses sparse_output=False
    - Older sklearn uses sparse=False
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    X: pd.DataFrame,
    use_onehot: bool,
    use_norm: bool,
    scaler_type: str
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Build a ColumnTransformer that:
    - imputes missing numeric values (median), optionally scales them
    - imputes missing categorical values (most frequent), one-hot encodes them
    Returns:
      preprocessor, categorical_cols, numeric_cols
    """
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    transformers = []

    # Numeric pipeline: impute -> (optional) scale
    if num_cols:
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if use_norm:
            scaler = StandardScaler() if scaler_type == "standard" else MinMaxScaler()
            num_steps.append(("scaler", scaler))
        transformers.append(("num", Pipeline(steps=num_steps), num_cols))

    # Categorical pipeline: impute -> one-hot
    if cat_cols:
        if not use_onehot:
            raise ValueError(
                f"Categorical FEATURE columns detected: {cat_cols}\n"
                "These are part of the model inputs (X). Enable One-Hot Encoding or remove/convert them."
            )
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _safe_onehot())
        ])
        transformers.append(("cat", cat_pipe, cat_cols))

    # remainder="drop" means we only keep processed columns
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, cat_cols, num_cols


def get_models(cfg: TrainConfig) -> Dict[str, object]:
    """Create the selected sklearn estimators based on the GUI config."""
    models: Dict[str, object] = {}

    if cfg.use_perceptron:
        models["Perceptron"] = Perceptron(max_iter=1000, tol=1e-3, random_state=cfg.random_state)

    if cfg.use_mlp:
        mlp = cfg.mlp
        layers = int(mlp.hidden_layers)
        if layers not in (1, 2, 3, 4):
            raise ValueError("MLP hidden layer count must be between 1 and 4.")

        # Build tuple dynamically based on selected layer count
        all_sizes = [int(mlp.n1), int(mlp.n2), int(mlp.n3), int(mlp.n4)]
        hidden_sizes = tuple(all_sizes[:layers])

        # Validate only the used layers
        if any(s <= 0 for s in hidden_sizes):
            raise ValueError("MLP neuron counts must be positive integers for the selected layers.")

        models["MLP"] = MLPClassifier(
            hidden_layer_sizes=hidden_sizes,
            activation=mlp.activation,
            learning_rate_init=float(mlp.learning_rate_init),
            max_iter=int(mlp.max_iter),
            random_state=cfg.random_state
        )

    if cfg.use_tree:
        models["Decision Tree"] = DecisionTreeClassifier(random_state=cfg.random_state)

    return models


def train_evaluate(df: pd.DataFrame, cfg: TrainConfig):
    """
    Main training function used by the GUI.
    Steps:
      1) Split df into X (features) and y (target)
      2) Optional: bin numeric target into classes (classification-friendly)
      3) Encode y if it is categorical/text
      4) Build preprocessing pipeline
      5) Train each selected model, compute metrics + confusion matrix
    Returns:
      results: list of (model_name, acc, precision, recall, f1)
      cm_store: dict model_name -> (confusion_matrix, class_labels)
      meta: small info dict used for logging (num_cols, cat_cols, etc.)
    """
    if cfg.target_col not in df.columns:
        raise ValueError("Target column not found in dataset.")

    # Separate features and label
    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col].copy()

    # Drop rows where target is missing (simpler + avoids training issues)
    mask = y.notna()
    X = X.loc[mask].copy()
    y = y.loc[mask].copy()

    # If target is numeric and user enabled binning, discretize it into k bins/classes
    if cfg.bin_numeric_target and pd.api.types.is_numeric_dtype(y):
        k = int(cfg.bin_count)
        if k < 2:
            raise ValueError("bin_count must be >= 2")

        uniq = int(y.nunique())
        if uniq < 2:
            raise ValueError("Target has < 2 unique values; classification is not possible.")

        # If there are fewer unique values than bins, reduce bins
        k = min(k, uniq)

        if cfg.bin_strategy == "quantile":
            # Equal-frequency bins (each bin has ~same number of samples)
            y_binned = pd.qcut(y, q=k, duplicates="drop")
        else:
            # Equal-width bins (bins cover equal numeric ranges)
            y_binned = pd.cut(y, bins=k)

        # Convert intervals to clean string labels for display
        y = y_binned.astype(str)

    # Encode string/categorical targets into integers (sklearn expects numeric class labels)
    label_encoder = None
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y.astype(str))

    # Build preprocessing for X (handles missing values + onehot/scaling)
    preprocessor, cat_cols, num_cols = build_preprocessor(
        X,
        use_onehot=cfg.use_onehot,
        use_norm=cfg.use_norm,
        scaler_type=cfg.scaler_type
    )

    # Build selected models
    models = get_models(cfg)
    if not models:
        raise ValueError("Select at least one model.")

    # Train/test split (stratify helps keep class distribution similar in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg.test_size),
        random_state=int(cfg.random_state),
        stratify=y if len(np.unique(y)) > 1 else None
    )

    results = []
    cm_store = {}

    # Train each selected model inside a pipeline: preprocess -> model
    for name, estimator in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", estimator)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Metrics required by the project (weighted = works for multi-class)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Confusion matrix for the selected model
        cm = confusion_matrix(y_test, y_pred)

        # Label names used on the confusion matrix axis
        if label_encoder is not None:
            labels = list(label_encoder.classes_)
        else:
            labels = [str(c) for c in sorted(np.unique(np.concatenate([y_test, y_pred])))]

        results.append((name, acc, prec, rec, f1))
        cm_store[name] = (cm, labels)

    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "label_encoder": label_encoder
    }
    return results, cm_store, meta
