# model/stacking.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from src.models import get_lgb_model, get_xgb_model


def train_stacking_with_holdout(X, y, n_splits=5, random_state=42):
    """
    Обучает стекинг с честной оценкой на hold-out.
    Возвращает: meta_model, base_models, X_test, y_test, test_pred_proba
    """
    # 🔑 1. Hold-out split: отделяем настоящий тест (20%)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.01, stratify=y, random_state=random_state
    )

    # 🔑 2. Генерация OOF-предсказаний на X_train_full с помощью StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_lgbm = np.zeros(len(X_train_full))
    oof_xgb = np.zeros(len(X_train_full))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        X_tr, X_val = X_train_full.iloc[tr_idx], X_train_full.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[tr_idx], y_train_full.iloc[val_idx]

        # Обучаем LGBM (возвращает Booster → .predict() даёт вероятности)
        lgb_model = get_lgb_model(X_tr, X_val, y_tr, y_val)
        oof_lgbm[val_idx] = lgb_model.predict(X_val)  # ← уже вероятности!

        # Обучаем XGBoost (возвращает XGBClassifier → .predict_proba()[:, 1])
        xgb_model = get_xgb_model(X_tr, X_val, y_tr, y_val)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]

    # 🔑 3. Обучаем мета-модель (логистическая регрессия) на OOF-признаках
    meta_features_train = np.column_stack([oof_lgbm, oof_xgb])
    meta_model = LogisticRegression(
        class_weight='balanced',
        random_state=random_state,
        max_iter=1000
    )
    meta_model.fit(meta_features_train, y_train_full)

    # 🔑 4. Обучаем финальные базовые модели на ВСЕМ X_train_full
    # Используем внутренний валидационный сплит только для early stopping
    X_tr_final, X_val_final, y_tr_final, y_val_final = train_test_split(
        X_train_full, y_train_full, test_size=0.01, stratify=y_train_full, random_state=random_state
    )
    final_lgb = get_lgb_model(X_tr_final, X_val_final, y_tr_final, y_val_final)
    final_xgb = get_xgb_model(X_tr_final, X_val_final, y_tr_final, y_val_final)

    # 🔑 5. Предсказание на hold-out (X_test)
    test_meta_features = np.column_stack([
        final_lgb.predict(X_test),               # ← LGBM: .predict() → вероятности
        final_xgb.predict_proba(X_test)[:, 1]    # ← XGBoost: .predict_proba()[:, 1]
    ])
    test_pred_proba = meta_model.predict_proba(test_meta_features)[:, 1]

    # Возвращаем всё необходимое
    base_models = {'lgbm': final_lgb, 'xgboost': final_xgb}
    return meta_model, base_models, X_test, y_test, test_pred_proba