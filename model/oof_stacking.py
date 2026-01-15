"""
Назначение:
Альтернативная реализация стекинга через OOF без hold-out.
Проблема:

Не вызывается ни из main.py, ни из других скриптов.
Использует собственную вспомогательную функцию train_test_split_for_early_stopping.
Статус: Старая версия
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

def oof_stacking(
    X: pd.DataFrame,
    y: pd.Series,
    model_types: list = ['lgbm', 'xgboost'],
    n_splits: int = 5,
    random_state: int = 42,
    meta_model_type: str = 'logistic'
) -> Tuple[Any, Dict[str, Any], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Стекинг с OOF для бинарной классификации.

    Возвращает:
        meta_model          — обученная мета-модель
        base_models         — словарь моделей, обученных на полном train
        oof_base_preds      — {model_type: oof_proba_array}
        oof_meta_preds      — предсказания мета-модели на OOF
        y_true              — истинные метки
    """
    from src.model_lgb import get_lgb_model
    from src.model_xgb import get_xgb_model

    # === 1. Обучение базовых моделей на полном train (для инференса) ===
    print("🔧 Обучение базовых моделей на полном train...")
    base_models = {}
    X_train_full, X_val_full, y_train_full, y_val_full = \
        train_test_split_for_early_stopping(X, y, test_size=0.4, random_state=random_state)

    for model_type in model_types:
        print(f"  → {model_type}")
        if model_type == 'lgbm':
            base_models[model_type] = get_lgb_model(X_train_full, X_val_full, y_train_full, y_val_full)
        elif model_type == 'xgboost':
            base_models[model_type] = get_xgb_model(X_train_full, X_val_full, y_train_full, y_val_full)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    # === 2. Генерация OOF-предсказаний через StratifiedKFold ===
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_samples = len(y)
    oof_preds = {m: np.zeros(n_samples) for m in model_types}

    print(f"🔄 Генерация OOF-предсказаний ({n_splits}-fold CV)...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Делим train на train/val для early stopping
        X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = \
            train_test_split_for_early_stopping(X_tr, y_tr, test_size=0.4, random_state=fold)

        for model_type in model_types:
            if model_type == 'lgbm':
                temp_model = get_lgb_model(X_tr_sub, X_val_sub, y_tr_sub, y_val_sub)
                oof_preds[model_type][val_idx] = temp_model.predict(X_val_fold)
            elif model_type == 'xgboost':
                temp_model = get_xgb_model(X_tr_sub, X_val_sub, y_tr_sub, y_val_sub)
                oof_preds[model_type][val_idx] = temp_model.predict_proba(X_val_fold)[:, 1]

    # === 3. Обучение мета-модели ===
    meta_features = np.column_stack([oof_preds[m] for m in model_types])

    if meta_model_type == 'logistic':
        meta_model = LogisticRegression(
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000
        )
    else:
        raise ValueError(f"Unsupported meta_model_type: {meta_model_type}")

    meta_model.fit(meta_features, y)
    oof_meta_preds = meta_model.predict_proba(meta_features)[:, 1]

    # === 4. Оценка ===
    auc_oof = roc_auc_score(y, oof_meta_preds)
    pr_auc_oof = average_precision_score(y, oof_meta_preds)
    print(f"\n✅ OOF ROC-AUC (стекинг): {auc_oof:.4f}")
    print(f"✅ OOF PR-AUC (стекинг):  {pr_auc_oof:.4f}")

    if hasattr(meta_model, 'coef_'):
        coef = meta_model.coef_[0]
        names = model_types
        print(f"  📈 Веса моделей: " + ", ".join([f"{n}={c:.3f}" for n, c in zip(names, coef)]))

    return (
        meta_model,
        base_models,
        oof_preds,
        oof_meta_preds,
        y.values
    )

def train_test_split_for_early_stopping(X, y, test_size=0.2, random_state=42):
    """Вспомогательная функция для разделения на train/val с сохранением стратификации."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)