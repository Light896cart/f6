"""
Назначение:
Главный скрипт запуска:

Загружает данные.
Вызывает train_stacking_with_holdout.
Оценивает на hold-out и на внешнем тесте.
Статус: Основной entry point.

"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.features import preprocess_features
from src.metrics import evaluate_model
from model.stacking import train_stacking_with_holdout


def main():
    # --- Пути к данным ---
    train_path = Path(r"train_dataset_6.csv")
    test_path = Path("test_dataset_6.csv")  # ← добавили путь к тесту

    if not train_path.exists():
        train_path = Path("data") / "android_device_info.csv"
        # Предположим, что если нет train/test, то и test нет → генерировать нельзя
        raise FileNotFoundError(f"Не найден train_dataset.csv и нет исходного датасета для разделения")

    # --- Файл с пакетами (если нужен) ---
    packages_path = Path("android_packages.csv.gz")
    if not packages_path.exists():
        packages_path = Path("data") / "android_packages.csv.gz"

    # === 1. Загрузка и препроцессинг TRAIN ===
    df_train = pd.read_csv(train_path)
    # df_train = preprocess_features(df_train, packages_path=str(packages_path))

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    # One-hot encoding (сохраним колонки для теста!)
    X_train = pd.get_dummies(X_train, dtype=int)
    train_columns = X_train.columns.tolist()  # ← запоминаем порядок и состав признаков

    # === 2. Обучение модели ===
    print("🚀 Запуск стекинга с hold-out оценкой...")
    meta_model, base_models, X_holdout, y_holdout, holdout_pred_proba = train_stacking_with_holdout(X_train, y_train)
    evaluate_model(y_holdout, holdout_pred_proba, "✅ HOLD-OUT Validation")

    # === 3. Загрузка и препроцессинг TEST ===
    if not test_path.exists():
        print("⚠️ Тестовый файл test_dataset.csv не найден. Оценка на тесте пропущена.")
        return

    df_test = pd.read_csv(test_path)
    # df_test = preprocess_features(df_test, packages_path=str(packages_path))

    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    # Применяем тот же one-hot encoding: выравниваем колонки по train
    X_test = pd.get_dummies(X_test, dtype=int)

    # Гарантируем, что в тесте есть все колонки из трейна (и только они)
    for col in train_columns:
        if col not in X_test.columns:
            X_test[col] = 0  # добавляем недостающие фичи как нули

    X_test = X_test[train_columns]  # сохраняем порядок как в трейне

    # === 4. Предсказание на тесте (точно как при создании hold-out предсказаний) ===
    print("🔮 Делаем предсказание на тестовом наборе...")

    # Получаем мета-признаки ТОЧНО ТАК ЖЕ, как в train_stacking_with_holdout
    lgbm_test_pred = base_models['lgbm'].predict(X_test)  # классы (0/1)
    xgb_test_pred = base_models['xgboost'].predict_proba(X_test)[:, 1]  # вероятности

    X_meta_test = np.column_stack([lgbm_test_pred, xgb_test_pred])
    test_pred_proba = meta_model.predict_proba(X_meta_test)[:, 1]

    # === 5. Оценка на тесте ===
    evaluate_model(y_test, test_pred_proba, "🎯 FINAL TEST SET Evaluation")


if __name__ == "__main__":
    main()