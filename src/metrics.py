"""
Назначение:
Полноценная оценка модели: от ROC-AUC до бизнес-метрик (стоимость FP/FN, экономия vs baseline).
Статус: Актуальный — используется в main.py.
"""
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix,
    matthews_corrcoef, brier_score_loss
)


def geometric_mean_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sensitivity * specificity)


def evaluate_model(y_true, y_pred_proba, model_name="Model",
                   cost_fp=1200, cost_fn=5000, print_business_metrics=True):
    """
    Оценивает модель по множеству метрик и рассчитывает бизнес-убытки.

    Параметры:
        y_true (array): истинные метки (0 — Genuine, 1 — Fraud)
        y_pred_proba (array): предсказанные вероятности класса Fraud
        model_name (str): название модели для вывода
        cost_fp (float): стоимость ложноположительного решения (FP), в рублях
        cost_fn (float): стоимость ложноотрицательного решения (FN), в рублях
        print_business_metrics (bool): выводить ли бизнес-метрики
    """
    # Используем порог 0.3, как в оригинале (можно параметризовать позже)
    y_pred = (y_pred_proba >= 0.4).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "gmean": geometric_mean_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_pred_proba)
    }

    print(f"\n📊 === {model_name} ===")
    for k, v in metrics.items():
        print(f"{k.replace('_', ' ').title():<20}: {v:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Genuine", "Fraud"]))

    cm = confusion_matrix(y_true, y_pred)
    print("\n🧮 Confusion Matrix:")
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"  TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    # === Бизнес-метрика ===
    if print_business_metrics:
        total_business_loss = fp * cost_fp + fn * cost_fn
        n_samples = len(y_true)
        avg_loss_per_sample = total_business_loss / n_samples

        print(f"\n💼 === Business Impact (Russia, 2025 Hypotheses) ===")
        print(f"Cost per FP (block legit user) : {cost_fp:,} ₽")
        print(f"Cost per FN (miss fraudster)    : {cost_fn:,} ₽")
        print(f"Total FP                        : {fp}")
        print(f"Total FN                        : {fn}")
        print(f"💰 Total Expected Business Loss : {total_business_loss:,.0f} ₽")
        print(f"📉 Avg Loss per Prediction      : {avg_loss_per_sample:.2f} ₽")

        # Дополнительно: сколько "спасено" по сравнению с baseline?
        # Например, если бы мы не использовали модель и пропускали всех → FN = total_fraud, FP = 0
        total_fraud = fn + tp
        loss_no_model = total_fraud * cost_fn  # все мошенники прошли
        savings = loss_no_model - total_business_loss
        avg_savings_per_sample = savings / n_samples

        print(f"\n💡 Estimated Savings vs No Model: {savings:,.0f} ₽")
        print(f"📈 Avg Savings per Sample       : {avg_savings_per_sample:.2f} ₽")

        if savings > 0:
            print("✅ Model provides positive business value.")
        else:
            print("⚠️  Model may be too aggressive or ineffective — consider tuning threshold.")

    return metrics