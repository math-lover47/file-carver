import numpy as np
import json
import os
from tqdm import tqdm

def create_lean_dataset():
    # Пути к исходным данным
    data_dir = "/home/arsen/file-carver"
    x_path = os.path.join(data_dir, "napier_X.npy")
    y_path = os.path.join(data_dir, "napier_Y.npy")
    l_path = os.path.join(data_dir, "napier_L.npy")
    json_path = os.path.join(data_dir, "classes.json")

    # Пути к новым данным
    output_x = os.path.join(data_dir, "napier_X_lean.npy")
    output_y = os.path.join(data_dir, "napier_Y_lean.npy")
    output_l = os.path.join(data_dir, "napier_L_lean.npy")
    output_json = os.path.join(data_dir, "classes_lean.json")

    # 1. Загружаем старые классы
    with open(json_path, 'r') as f:
        old_classes = json.load(f)
    
    # Список классов, которые мы ХОТИМ оставить (примерно 40 типов)
    # Убираем дубликаты компрессии и качества
    keep_list = [
        "PDF", "DOCX", "XLSX", "PPTX", "TXT", "RTF", "JSON", "XML", "CSV", "HTML", "JAVASCRIPT",
        "JPG-q075", "PNG-c5", "GIF", "BMP", "MP4", "MKV", "MP3", "WAV", "TIF",
        "ZIP-DEFLATE", "7ZIP-LZMA", "TAR", "RAR", "GZIP",
        "EXE", "DLL", "ELF", "APK", "MSI",
        "SQLITE", "PCAP", "DWG", "EPUB",
        "RANSOMWARE-RYUK", "RANSOMWARE-MAZE", # Оставим пару для примера
        "lossless-c4", "q50-c4" # Оставим по одному из спец-форматов
    ]
    
    # Маппинг имен в старые ID
    name_to_old_id = {name: idx for name, idx in old_classes.items()}
    allowed_old_ids = []
    for name in keep_list:
        if name in name_to_old_id:
            allowed_old_ids.append(name_to_old_id[name])
    
    # 2. Загружаем метаданные (Y и L) в RAM
    print("Загрузка метаданных в память...")
    Y = np.load(y_path)
    L = np.load(l_path)
    X_mmap = np.load(x_path, mmap_mode='r')
    
    # 3. Фильтрация индексов
    print("Фильтрация индексов (Stratified Sampling)...")
    selected_indices = []
    
    # Словари для лимитов
    class_counts = {cls_id: 0 for cls_id in allowed_old_ids}
    file_block_counts = {} # file_id -> count
    
    MAX_PER_CLASS = 50000  # Макс блоков на тип файла
    MAX_PER_FILE = 1000    # Макс блоков из одного файла (чтобы не перегружать MP4/MKV)
    
    for i in tqdm(range(len(Y))):
        cls_id = Y[i]
        if cls_id not in class_counts:
            continue
            
        if class_counts[cls_id] >= MAX_PER_CLASS:
            continue
            
        file_id = L[i, 0]
        if file_id not in file_block_counts:
            file_block_counts[file_id] = 0
            
        if file_block_counts[file_id] >= MAX_PER_FILE:
            continue
            
        selected_indices.append(i)
        class_counts[cls_id] += 1
        file_block_counts[file_id] += 1
    
    selected_indices = np.array(selected_indices)
    print(f"Выбрано блоков: {len(selected_indices)} (из {len(Y)})")
    
    # 4. Создаем новый маппинг классов (чтобы ID шли подряд 0, 1, 2...)
    new_classes_config = {}
    old_id_to_new_id = {}
    current_new_id = 0
    
    # Сортируем по именам для красоты
    for name in sorted(keep_list):
        if name in name_to_old_id:
            old_id = name_to_old_id[name]
            new_classes_config[name] = current_new_id
            old_id_to_new_id[old_id] = current_new_id
            current_new_id += 1
            
    # 5. Сохраняем новые данные
    print("Сохранение нового датасета (это может занять 5-10 минут)...")
    
    # Новые Y (метки)
    new_Y = np.array([old_id_to_new_id[Y[i]] for i in selected_indices], dtype=np.uint8)
    np.save(output_y, new_Y)
    
    # Новые L (ссылки)
    new_L = L[selected_indices]
    np.save(output_l, new_L)
    
    # Новые X (байты) - читаем пачками, чтобы не убить RAM
    new_X = np.zeros((len(selected_indices), 512), dtype=np.uint8)
    batch_size = 10000
    for start in tqdm(range(0, len(selected_indices), batch_size), desc="Копирование X"):
        end = start + batch_size
        batch_indices = selected_indices[start:end]
        new_X[start:end] = X_mmap[batch_indices]
        
    np.save(output_x, new_X)
    
    # Новый JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(new_classes_config, f, indent=4)
        
    print("\nГотово!")
    print(f"Новый размер X: {new_X.shape}")
    print(f"Новый JSON сохранен в {output_json}")
    print("Теперь вы можете использовать файлы с припиской '_lean' для обучения.")

if __name__ == "__main__":
    create_lean_dataset()
