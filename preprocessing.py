########## README
# Этот README нужен, так как при загрузке решений на платформе нельзя указать его. Поэтому, пожалуйста, прочтите.
# Установите нужные версии библиотек:
"""
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install numpy==2.1.2 pandas==2.3.1 opencv-python==4.12.0.88 matplotlib==3.10.5 tqdm==4.67.1 
"""

# Для работы скрипта нужно запускать все ########## README
# Этот README нужен, так как при загрузке решений на платформе нельзя указать его. Поэтому, пожалуйста, прочтите.
# Установите нужные версии библиотек:
"""
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
pip install numpy==2.1.2 pandas==2.3.1 opencv-python==4.12.0.88 matplotlib==3.10.5 tqdm==4.67.1 
"""

# Для работы скрипта нужно запускать все в порядке:
# 1) preprocessing.py
# 2) train.py
# 3) test.py 

# В директории с файлами должны находится файлы: 
# - train_histograms.zip
# - train_imgs.zip 
# - test_imgs.zip 
# - test.csv
# - train.csv 
# - metadata.csv 

# Рекомендуемые версии python: 3.12.11/3.11.13
##########в порядке:
# 1) preprocessing.py
# 2) train.py
# 3) test.py 

# В директории с файлами должны находится файлы: 
# - train_histograms.zip
# - train_imgs.zip 
# - test_imgs.zip 
# - test.csv
# - train.csv 
# - metadata.csv 

# Рекомендуемые версии python: 3.12.11/3.11.13
##########

import os
import torch
import zipfile
import shutil

print('Проверка окружения и директории')

required_files = ['train_histograms.zip', 'train_imgs.zip', 'test_imgs.zip', 'test.csv', 'train.csv', 'metadata.csv']

for file in required_files:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Не найден обязательный файл: {file}")

print("Все необходимые файлы найдены в директориию. 1/3 проверок пройдена.")
print(torch.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("Требуется GPU T4, но CUDA недоступно")

device_name = torch.cuda.get_device_name(0)

if "T4" not in device_name:
    raise RuntimeError(f"Требуется GPU T4, но обнаружен: {device_name}")
    
print(f"Используется GPU T4: {device_name}. 2/3 проверок пройдена.")

required_versions = {
    'torch': '2.5.1+cu124',
    'torchvision': '0.20.1+cu124', 
    'numpy': '2.1.2'
}

for lib, required_version in required_versions.items():
    try:
        current_version = __import__(lib).__version__
        if current_version != required_version:
            raise RuntimeError(f"Требуется {lib} версии {required_version}, но установлена {current_version}")
        print(f"Версия {lib}: {current_version} - корректна.")
    except ImportError:
        raise ImportError(f"Библиотека {lib} не установлена")

print("Все версии библиотек корректны. 3/3 проверок пройдена.")

zip_files = ['train_histograms.zip', 'train_imgs.zip', 'test_imgs.zip']

for zip_file in zip_files:
    print(f"Распаковка {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()
    print(f"{zip_file} распакован")

print("Все zip файлы распакованы")

# Удаляем дубликаты папок

def fix_duplicated_folder(main_folder):
    inner_folder = os.path.join(main_folder, os.path.basename(main_folder))

    if not os.path.exists(inner_folder) or not os.path.isdir(inner_folder):
        print(f"Внутренняя папка {inner_folder} не существует")
        return

    # Перемещаем все содержимое внутренней папки
    for item in os.listdir(inner_folder):
        src_path = os.path.join(inner_folder, item)
        dst_path = os.path.join(main_folder, item)

        # Если элемент уже существует в основной папке
        if os.path.exists(dst_path):
            if os.path.isdir(src_path):
                # Для папок: объединяем содержимое
                for sub_item in os.listdir(src_path):
                    sub_src = os.path.join(src_path, sub_item)
                    sub_dst = os.path.join(dst_path, sub_item)
                    if os.path.exists(sub_dst):
                        shutil.rmtree(sub_dst) if os.path.isdir(sub_dst) else os.remove(sub_dst)
                    shutil.move(sub_src, sub_dst)
                shutil.rmtree(src_path)
            else:
                # Для файлов: заменяем
                os.remove(dst_path)
                shutil.move(src_path, dst_path)
        else:
            shutil.move(src_path, dst_path)

    # Удаляем пустую внутреннюю папку
    if not os.listdir(inner_folder):
        shutil.rmtree(inner_folder)
        print(f"Папка {inner_folder} успешно удалена")
    else:
        print(f"Не удалось полностью очистить папку {inner_folder}")

# Пример использования
fix_duplicated_folder('train_imgs')
fix_duplicated_folder('test_imgs')
fix_duplicated_folder('train_histograms')

assert len(os.listdir('test_imgs')) == 145, f"В test_imgs должно быть 145 изображений, но найдено {len(os.listdir('test_imgs'))}"
assert len(os.listdir('train_imgs')) == 570, f"В train_imgs должно быть 570 изображений, но найдено {len(os.listdir('train_imgs'))}"