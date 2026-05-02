import os
import json
import numpy as np
from tqdm import tqdm


def create_micro_dataset(lean_dir, output_dir, target_classes):
    print(f"Loading lean dataset from {lean_dir}...")

    # 1. Загружаем старый словарь классов
    with open(os.path.join(lean_dir, "classes_lean.json"), "r") as f:
        classes_lean = json.load(f)

    # Определяем структуру словаря (бывает "0": "jpeg" или "jpeg": 0)
    first_key = str(next(iter(classes_lean.keys())))
    if first_key.isdigit():
        name_to_old_id = {name: int(idx) for idx, name in classes_lean.items()}
    else:
        name_to_old_id = {name: int(idx) for name, idx in classes_lean.items()}

    valid_targets = []
    old_id_to_new_id = {}
    new_classes_dict = {}

    # 2. Создаем новые ID (от 0 до N-1)
    new_id = 0
    for name in target_classes:
        if name in name_to_old_id:
            old_id = name_to_old_id[name]
            valid_targets.append(old_id)
            old_id_to_new_id[old_id] = new_id
            new_classes_dict[new_id] = name
            new_id += 1
        else:
            print(f"⚠️ Warning: Class '{name}' not found in lean dataset.")

    print(
        f"✅ Selected {len(valid_targets)} classes: {list(new_classes_dict.values())}")

    os.makedirs(output_dir, exist_ok=True)

    # Сохраняем новые классы
    with open(os.path.join(output_dir, "classes_micro.json"), "w") as f:
        json.dump(new_classes_dict, f, indent=4)

    # 3. Фильтруем Y и сэмплируем по ФАЙЛАМ (целиком)
    print("Loading Y and L to sample by File ID...")
    Y_lean = np.load(os.path.join(lean_dir, "napier_Y_lean.npy"))
    L_lean = np.load(os.path.join(lean_dir, "napier_L_lean.npy"))
    
    MAX_FILES_PER_CLASS = 30 # 30 файлов (4-5 мин на эпоху)
    selected_indices = []
    
    np.random.seed(42)
    for old_id in valid_targets:
        class_mask = (Y_lean == old_id)
        class_fids = np.unique(L_lean[class_mask, 1])
        
        if len(class_fids) > MAX_FILES_PER_CLASS:
            selected_fids = np.random.choice(class_fids, MAX_FILES_PER_CLASS, replace=False)
        else:
            selected_fids = class_fids
            
        fid_mask = np.isin(L_lean[:, 1], selected_fids)
        final_mask = class_mask & fid_mask
        selected_indices.extend(np.where(final_mask)[0].tolist())
        
    indices = np.array(selected_indices)
    indices.sort()
    
    print(f"📉 Reduced dataset size from {len(Y_lean)} to {len(indices)} blocks by keeping full files.")

    print("Processing and saving Y_micro...")
    Y_micro = Y_lean[indices]
    for old_id, n_id in old_id_to_new_id.items():
        Y_micro[Y_micro == old_id] = n_id
    np.save(os.path.join(output_dir, "napier_Y_micro.npy"), Y_micro)

    # 4. Фильтруем L (метаданные блоков)
    print("Processing and saving L_micro...")
    L_micro = L_lean[indices]
    np.save(os.path.join(output_dir, "napier_L_micro.npy"), L_micro)

    # 5. Фильтруем X (сами байты)
    print("Processing and saving X_micro (might take a minute)...")
    # Используем mmap, чтобы не загружать 1.2 млн блоков (600 МБ) в память целиком
    X_lean = np.load(os.path.join(
        lean_dir, "napier_X_lean.npy"), mmap_mode='r')
    X_micro = np.empty((len(indices), 512), dtype=np.uint8)

    chunk_size = 10000
    for i in tqdm(range(0, len(indices), chunk_size)):
        chunk_idx = indices[i:i+chunk_size]
        X_micro[i:i+chunk_size] = X_lean[chunk_idx]

    np.save(os.path.join(output_dir, "napier_X_micro.npy"), X_micro)

    print("🚀 Micro dataset created successfully in:", output_dir)


if __name__ == "__main__":
    # --- НАСТРОЙКИ ---
    # Укажите путь к вашему ТЕКУЩЕМУ lean датасету (в Kaggle это папка в /kaggle/input)
    # Если запускаете скрипт локально, укажите локальный путь
    INPUT_DIR = "./"

    # Папка, куда сохранятся новые файлы
    OUTPUT_DIR = "./napier_micro"

    # Список классов, которые мы оставляем (самые популярные форматы)
    # Список классов, которые принципиально отличаются по структуре байтов
    TARGET_CLASSES = [
        "JPG-q075", "PDF", "PNG-c5", "TXT", "MP3"
    ]

    create_micro_dataset(INPUT_DIR, OUTPUT_DIR, TARGET_CLASSES)
