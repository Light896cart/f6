"""
Назначение:
Генерация профиля данных (пропуски, распределения, корреляции) → сохраняет в JSON.
Статус: Вспомогательный — полезен на этапе EDA, но не входит в inference pipeline.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any

def analyze_column(series: pd.Series) -> Dict[str, Any]:
    col_info = {}
    col_info["dtype"] = str(series.dtype)
    n_total = len(series)
    n_missing = series.isna().sum()
    col_info["n_missing"] = int(n_missing)
    col_info["pct_missing"] = float(n_missing / n_total * 100)
    col_info["n_unique"] = int(series.nunique(dropna=True))

    # Числовые признаки
    if pd.api.types.is_numeric_dtype(series):
        s_clean = series.dropna()
        if len(s_clean) == 0:
            # Если все NaN — заполняем нулями
            stats = {
                "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
                "skewness": 0.0, "kurtosis": 0.0, "q1": 0.0, "q2_median": 0.0, "q3": 0.0, "iqr": 0.0
            }
        else:
            q1 = float(s_clean.quantile(0.25))
            q2 = float(s_clean.median())
            q3 = float(s_clean.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers_iqr = int(((s_clean < lower_bound) | (s_clean > upper_bound)).sum())

            z_scores = np.abs((s_clean - s_clean.mean()) / s_clean.std()) if s_clean.std() != 0 else np.zeros_like(s_clean)
            outliers_zscore = int((z_scores > 3).sum())

            percentiles = {
                "p1": float(s_clean.quantile(0.01)),
                "p5": float(s_clean.quantile(0.05)),
                "p10": float(s_clean.quantile(0.10)),
                "p25": q1,
                "p50": q2,
                "p75": q3,
                "p90": float(s_clean.quantile(0.90)),
                "p95": float(s_clean.quantile(0.95)),
                "p99": float(s_clean.quantile(0.99)),
            }

            stats = {
                "mean": float(s_clean.mean()),
                "median": q2,
                "std": float(s_clean.std()),
                "min": float(s_clean.min()),
                "max": float(s_clean.max()),
                "skewness": float(s_clean.skew()),
                "kurtosis": float(s_clean.kurtosis()),
                "q1": q1,
                "q2_median": q2,
                "q3": q3,
                "iqr": iqr,
                "outliers_iqr": outliers_iqr,
                "outliers_zscore": outliers_zscore,
                "percentiles": percentiles
            }
        col_info.update(stats)

    # Категориальные / объектные признаки
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        value_counts = series.value_counts(dropna=False)
        if len(value_counts) > 0:
            most_frequent_value = value_counts.index[0]
            most_frequent_count = int(value_counts.iloc[0])
            most_frequent_pct = float(most_frequent_count / n_total * 100)
            top_5 = {str(k): int(v) for k, v in value_counts.head(5).items()}
            sample_vc = {str(k): int(v) for k, v in value_counts.head(3).items()}
        else:
            most_frequent_value = None
            most_frequent_count = 0
            most_frequent_pct = 0.0
            top_5 = {}
            sample_vc = {}

        col_info.update({
            "most_frequent_value": str(most_frequent_value) if pd.notna(most_frequent_value) else "NaN",
            "most_frequent_count": most_frequent_count,
            "most_frequent_pct": most_frequent_pct,
            "top_5_values": top_5,
            "value_counts_sample": sample_vc
        })

    return col_info

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    columns_info = {}
    for col in df.columns:
        columns_info[col] = analyze_column(df[col])

    # Корреляция только для числовых колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_matrix = {}
    highly_correlated_pairs = []

    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr(method="pearson")
        # Округляем до 3 знаков
        corr_rounded = corr.round(3)
        corr_matrix = corr_rounded.to_dict()

        # Находим сильно коррелирующие пары (>0.9 или <-0.9), исключая диагональ
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                val = corr.iloc[i, j]
                if abs(val) >= 0.9:
                    highly_correlated_pairs.append({
                        "col1": numeric_cols[i],
                        "col2": numeric_cols[j],
                        "correlation": float(val)
                    })

    return {
        "columns": columns_info,
        "correlation": {
            "highly_correlated_pairs": highly_correlated_pairs,
            "correlation_matrix_sample": {k: v for k, v in corr_matrix.items() if k in numeric_cols[:5]}  # первые 5
        }
    }

def main():
    data_path = Path("train_dataset.csv")
    if not data_path.exists():
        data_path = Path("data") / "android_device_info.csv"

    print(f"Загружаем данные из: {data_path}")
    df = pd.read_csv(data_path)

    print("Анализируем датасет...")
    profile = analyze_dataset(df)

    output_path = Path("data_profile_with_new_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"✅ Профиль сохранён в: {output_path}")

if __name__ == "__main__":
    main()