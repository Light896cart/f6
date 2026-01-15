"""Назначение:
Создаёт сбалансированный 50/50 датасет.
Статус: Утилита — может использоваться для экспериментов, но в основном коде балансировка делается через class_weight='balanced'.
"""

import pandas as pd

# Читаем файл
df = pd.read_csv("D:/Code/test_task_f6/test_task_f6/train_dataset_6.csv")

# Проверка: есть ли данные?
if df.empty:
    raise ValueError("❌ Файл пустой!")

# Проверяем колонку target
if 'target' not in df.columns:
    raise KeyError("❌ Нет колонки 'target'")

# Считаем значения
counts = df['target'].value_counts().sort_index()
print("Распределение классов:")
print(counts)

# Определяем, есть ли оба класса: 0 и 1
if 0 not in counts or 1 not in counts:
    raise ValueError("❌ Один из классов (0 или 1) отсутствует. Балансировка невозможна.")

n_0 = counts[0]  # genuine
n_1 = counts[1]  # fraud
n = min(n_0, n_1)

print(f"Будет выбрано по {n} записей из каждого класса (0 и 1).")

# Выбираем сэмплы
class_0 = df[df['target'] == 0].sample(n=n, random_state=42)
class_1 = df[df['target'] == 1].sample(n=n, random_state=42)

# Объединяем и перемешиваем
balanced = pd.concat([class_0, class_1], ignore_index=True)
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Сохраняем
output_path = "D:/Code/test_task_f6/test_task_f6/train_dataset_balanced_50_50.csv"
balanced.to_csv(output_path, index=False)

print(f"✅ Успешно сохранено {len(balanced)} строк (0: {n}, 1: {n}) в {output_path}")