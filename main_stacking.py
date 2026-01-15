"""
Назначение:
Упрощённый стекинг без hold-out, с обычным train/test split.
Проблема:

Использует get_lgb_model() без аргументов → вызовет ошибку (функция требует X_train, X_val...).
Не согласуется с остальной архитектурой (нет early stopping через val set).
Статус: Устарел"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import dump

from src.features import preprocess_features
from src.model_lgb import get_lgb_model
from src.model_xgb import get_xgb_model
from src.metrics import evaluate_model

def main():
    # Загрузка данных
    data_path = Path("android_device_info.csv")
    if not data_path.exists():
        data_path = Path("data") / "android_device_info.csv"
    df = pd.read_csv(data_path)
    df = preprocess_features(df)

    X = df.drop(columns=["target"])
    y = df["target"]

    # One-Hot кодирование
    X = pd.get_dummies(X, dtype=int)

    # Разделение
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # === Обучение базовых моделей ===
    print("Обучаем LightGBM...")
    lgb_model = get_lgb_model()
    lgb_model.fit(X_train, y_train)
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]

    print("Обучаем XGBoost...")
    xgb_model = get_xgb_model()
    xgb_model.fit(X_train, y_train)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

    # === Оценка базовых моделей ===
    evaluate_model(y_test, lgb_proba, "LightGBM")
    evaluate_model(y_test, xgb_proba, "XGBoost")

    # === Стекинг: формируем матрицу предсказаний ===
    stack_train = np.column_stack([
        lgb_model.predict_proba(X_train)[:, 1],
        xgb_model.predict_proba(X_train)[:, 1]
    ])
    stack_test = np.column_stack([lgb_proba, xgb_proba])

    # Масштабирование (не обязательно для логрегрессии, но полезно)
    scaler = StandardScaler()
    stack_train_scaled = scaler.fit_transform(stack_train)
    stack_test_scaled = scaler.transform(stack_test)

    # Meta-learner
    print("\nОбучаем meta-learner (Logistic Regression)...")
    meta_model = LogisticRegression(
        class_weight="balanced",
        random_state=42,
        max_iter=1000
    )
    meta_model.fit(stack_train_scaled, y_train)
    stacking_proba = meta_model.predict_proba(stack_test_scaled)[:, 1]

    # === Оценка стекинга ===
    evaluate_model(y_test, stacking_proba, "Stacking (LGBM + XGBoost)")

    # Сохраняем модели (опционально)
    # dump(lgb_model, "models/lgb_model.joblib")
    # dump(xgb_model, "models/xgb_model.joblib")
    # dump(meta_model, "models/stacking_meta.joblib")
    # dump(scaler, "models/stacking_scaler.joblib")

    print("\n✅ Стекинг завершён!")

if __name__ == "__main__":
    main()