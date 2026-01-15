"""
Назначение:
Главный скрипт запуска с поддержкой двух режимов:

- mode="preprocessed": использует train_dataset_6.csv и test_dataset_6.csv напрямую.
- mode="raw": обрабатывает СЫРОЙ train (android_device_info.csv) → train_dataset_6.csv,
              но тест БЕРЁТСЯ ГОТОВЫЙ из test_dataset_6.csv (не обрабатывается заново!).
- mode="auto": как preprocessed, если оба файла есть; иначе — raw (но тест всё равно готовый).

❗ ВАЖНО: test_dataset_6.csv считается финальным валидационным/тестовым набором и НЕ пересоздаётся.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import joblib
from src.features import preprocess_features
from src.metrics import evaluate_model
from model.stacking import train_stacking_with_holdout


def main(mode: str = "auto", save_model: bool = True):
    # --- Пути ---
    processed_train = Path("train_dataset_6.csv")
    processed_test = Path("test_dataset_6.csv")
    raw_train_path = Path("android_device_info.csv")
    packages_path = Path("android_packages.csv.gz")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # === ВСЕГДА используем фиксированный тест ===
    if not processed_test.exists():
        raise FileNotFoundError(f"Тестовый файл {processed_test} ОБЯЗАТЕЛЕН!")
    df_test = pd.read_csv(processed_test)
    if "target" not in df_test.columns:
        raise ValueError(f"{processed_test} должен содержать 'target'.")

    # === Работа с трейном в зависимости от режима ===
    if mode == "preprocessed":
        print("🟢 Режим: Используем готовый train_dataset_6.csv")
        if not processed_train.exists():
            raise FileNotFoundError(f"Требуется {processed_train} в режиме 'preprocessed'")
        df_train = pd.read_csv(processed_train)


    elif mode == "raw":
        print("🟡 Режим: Обрабатываем СЫРОЙ трейн (android_device_info.csv) → train_dataset_6.csv")
        if not raw_train_path.exists():
            raise FileNotFoundError(f"Не найден сырой трейн: {raw_train_path}")
        if not packages_path.exists():
            raise FileNotFoundError(f"Не найден файл пакетов: {packages_path}")
        df_raw_train = pd.read_csv(raw_train_path)
        if "target" not in df_raw_train.columns or "agent_id" not in df_raw_train.columns:
            raise ValueError("Сырой трейн должен содержать 'target' и 'agent_id'.")
        # Обрабатываем ТОЛЬКО трейн
        df_train = preprocess_features(df_raw_train, packages_path=str(packages_path))

    elif mode == "auto":
        if processed_train.exists():
            print("🟢 Авто-режим: готовый трейн найден → используем его.")
            df_train = pd.read_csv(processed_train)
            print("🟢 Авто: готовый трейн найден")
        else:
            return main(mode="raw", save_model=save_model)
    else:
        raise ValueError("mode должен быть 'auto', 'preprocessed' или 'raw'")

    # === Проверка целевой переменной в трейне ===
    if "target" not in df_train.columns:
        raise ValueError("Трейн должен содержать колонку 'target'.")

    # === Общая логика обучения и оценки ===
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    X_train = pd.get_dummies(X_train, dtype=int)
    train_columns = X_train.columns.tolist()

    print("🚀 Запуск стекинга с hold-out оценкой...")
    meta_model, base_models, X_holdout, y_holdout, holdout_pred_proba = train_stacking_with_holdout(X_train, y_train)
    evaluate_model(y_holdout, holdout_pred_proba, "✅ HOLD-OUT Validation")

    # === Сохранение модели и колонок ===
    if save_model:
        joblib.dump(base_models, model_dir / "base_models.joblib")
        joblib.dump(meta_model, model_dir / "meta_model.joblib")
        joblib.dump(train_columns, model_dir / "train_columns.joblib")
        print(f"💾 Модель сохранена в {model_dir}/")

    # === Оценка на ФИКСИРОВАННОМ тесте ===
    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]
    X_test = pd.get_dummies(X_test, dtype=int)

    # Выравнивание колонок по трейну
    for col in train_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[train_columns]

    print("🔮 Делаем предсказание на фиксированном тестовом наборе (test_dataset_6.csv)...")
    lgbm_test_pred = base_models['lgbm'].predict(X_test)
    xgb_test_pred = base_models['xgboost'].predict_proba(X_test)[:, 1]
    X_meta_test = np.column_stack([lgbm_test_pred, xgb_test_pred])
    test_pred_proba = meta_model.predict_proba(X_meta_test)[:, 1]

    evaluate_model(y_test, test_pred_proba, "🎯 FINAL TEST SET Evaluation (test_dataset_6.csv)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML-пайплайн с фиксированным тестом")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "preprocessed", "raw"],
        default="preprocessed",
        help="Режим работы: auto (по умолчанию), preprocessed (готовые данные), raw (обработать сырой трейн)"
    )
    args = parser.parse_args()

    main(mode=args.mode)