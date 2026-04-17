import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(y_true, y_pred):
    """Расчет метрик"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1': f1_score(y_true, y_pred, average='binary')
    }


def print_metrics_table(mfcc_metrics, yamnet_metrics):
    """Вывод таблицы метрик"""
    print("\n" + "=" * 60)
    print(f"{'Метрика':<15} {'MFCC+':<20} {'YAMNet':<20}")
    print("=" * 60)

    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        m_val = mfcc_metrics[metric]
        y_val = yamnet_metrics[metric]
        winner = "✓" if m_val > y_val else ("✓" if y_val > m_val else "=")
        print(f"{metric.capitalize():<15} {m_val:.4f} {'':<15} {y_val:.4f} {'':<15} {winner}")

    print("=" * 60)


def get_confusion_matrix(y_true, y_pred):
    """Получение матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    return {
        'tn': cm[0, 0],  # True Negative
        'fp': cm[0, 1],  # False Positive
        'fn': cm[1, 0],  # False Negative
        'tp': cm[1, 1]  # True Positive
    }