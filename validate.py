"""
Назначение:
Оценка уже обученной модели на фиксированном тесте (test_dataset_6.csv).
Без обучения!
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import argparse
from src.metrics import evaluate_model


def validate(model_dir: str = "models"):
    model_dir = Path(model_dir)
    test_path = Path("test_dataset_6.csv")

    # Проверки
    if not test_path.exists():
        raise FileNotFoundError(f"Тест {test_path} не найден!")
    if not (model_dir / "base_models.joblib").exists():
        raise FileNotFoundError(f"Модель не найдена в {model_dir}")

    # Загрузка
    print("📥 Загружаем модель...")
    base_models = joblib.load(model_dir / "base_models.joblib")
    meta_model = joblib.load(model_dir / "meta_model.joblib")
    train_columns = joblib.load(model_dir / "train_columns.joblib")

    print("📥 Загружаем тест...")
    df_test = pd.read_csv(test_path)
    if "target" not in df_test.columns:
        raise ValueError("Тест должен содержать 'target'")

    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    # One-hot + выравнивание
    X_test = pd.get_dummies(X_test, dtype=int)
    for col in train_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[train_columns]

    # Предсказание
    print("🔮 Делаем предсказание...")
    lgbm_test_pred = base_models['lgbm'].predict(X_test)
    xgb_test_pred = base_models['xgboost'].predict_proba(X_test)[:, 1]
    X_meta_test = np.column_stack([lgbm_test_pred, xgb_test_pred])
    test_pred_proba = meta_model.predict_proba(X_meta_test)[:, 1]

    # Оценка
    evaluate_model(y_test, test_pred_proba, "🎯 VALIDATION ON FIXED TEST SET")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка модели без обучения")
    parser.add_argument("--model-dir", type=str, default="models", help="Папка с моделью")
    args = parser.parse_args()

    validate(model_dir=args.model_dir)