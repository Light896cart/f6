# src/thresholding.py
import numpy as np
from sklearn.metrics import fbeta_score, matthews_corrcoef, precision_recall_curve


def find_optimal_threshold(
        y_true,
        y_proba,
        metric="f2",  # или "mcc", "gmean", "custom"
        beta=2.0
):
    """
    Находит оптимальный порог бинаризации вероятностей.

    Args:
        y_true: истинные метки (0/1)
        y_proba: предсказанные вероятности класса 1 (Fraud)
        metric: 'f2', 'mcc', 'gmean', или callable
        beta: параметр для F-beta (beta>1 → акцент на recall)

    Returns:
        best_threshold: оптимальный порог
        best_score: значение метрики при этом пороге
    """
    if metric == "f2":
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -1
        best_threshold = 0.5
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = fbeta_score(y_true, y_pred, beta=beta)
            if score > best_score:
                best_score = score
                best_threshold = t
        return best_threshold, best_score

    elif metric == "mcc":
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -2
        best_threshold = 0.5
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = matthews_corrcoef(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_threshold = t
        return best_threshold, best_score

    elif metric == "gmean":
        from imblearn.metrics import geometric_mean_score
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -1
        best_threshold = 0.5
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = geometric_mean_score(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_threshold = t
        return best_threshold, best_score

    elif callable(metric):
        # Пользовательская функция: metric(y_true, y_pred) → float
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -np.inf
        best_threshold = 0.5
        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            score = metric(y_true, y_pred)
            if score > best_score:
                best_score = score
                best_threshold = t
        return best_threshold, best_score

    else:
        raise ValueError("Unsupported metric")