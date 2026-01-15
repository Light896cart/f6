"""
Назначение:
Делит android_device_info.csv на train/test (80/20).
Статус: Утилита.
"""


import pandas as pd

# Загрузка данных
df = pd.read_csv('android_device_info.csv')

# Перемешивание строк
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Разделение на 80% и 20%
split_index = int(0.8 * len(df_shuffled))
train_df = df_shuffled[:split_index]
test_df = df_shuffled[split_index:]

# Сохранение в два файла
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print("Файлы успешно разделены и сохранены.")