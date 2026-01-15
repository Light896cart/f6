"""
Назначение:
Распаковывает .gz → .csv.
Статус: Разовая утилита — не нужна, если данные уже в нужном формате.
"""
import gzip
import shutil

# Имя сжатого файла
input_file = "android_packages_test.csv.gz"
# Имя распакованного файла (без .gz)
output_file = "android_packages.csv"

# Распаковываем без удаления исходного .gz файла
with gzip.open(input_file, 'rb') as f_in:
    with open(output_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

print(f"Файл распакован как {output_file}. Оригинал {input_file} сохранён.")