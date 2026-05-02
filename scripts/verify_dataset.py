import numpy as np
import json
import os

def verify():
    # Пути к новым .npy файлам
    x_path = "napier_X.npy"
    y_path = "napier_Y.npy"
    l_path = "napier_L.npy"
    json_path = "classes.json"

    if not all(os.path.exists(p) for p in [x_path, y_path, l_path, json_path]):
        print(f"❌ Файлы не найдены! Сначала запустите обновленный process_napier.py.")
        return

    print("--- Загрузка данных (используя Memory Mapping) ---")
    # mmap_mode='r' позволяет работать с файлом на диске без загрузки в RAM
    X = np.load(x_path, mmap_mode='r')
    Y = np.load(y_path, mmap_mode='r')
    L = np.load(l_path, mmap_mode='r')

    with open(json_path, 'r') as f:
        classes = json.load(f)

    print(f"✅ Успешно подключено к данным.")
    print(f"Классов найдено: {len(classes)} ({', '.join(list(classes.keys())[:5])}...)")
    
    print("\n--- Проверка размерностей ---")
    n_blocks = X.shape[0]
    print(f"Количество блоков: {n_blocks}")
    print(f"Размер блока: {X.shape[1]} (ожидалось 512)")
    
    if n_blocks == len(Y) == len(L):
        print(f"✅ Длины массивов совпадают ({n_blocks}).")
    else:
        print(f"❌ ОШИБКА: Длины массивов различаются!")
        print(f"X: {len(X)}, Y: {len(Y)}, L: {len(L)}")

    print("\n--- Проверка типов данных ---")
    print(f"X dtype: {X.dtype} (ожидалось uint8)")
    print(f"Y dtype: {Y.dtype}")
    print(f"L dtype: {L.dtype}")

    print("\n--- Проверка структуры связей (links) ---")
    # Проверим первые 10 блоков
    print("Первые 5 записей [file_id, block_idx]:")
    print(L[:5])
    
    # Проверка последовательности индексов внутри одного файла
    unique_files = np.unique(L[:, 0])
    print(f"Всего уникальных файлов в датасете: {len(unique_files)}")
    
    # Берем случайный файл, у которого больше 1 блока
    file_mask = (L[:, 0] == unique_files[min(5, len(unique_files)-1)])
    file_blocks = L[file_mask]
    print(f"Индексы блоков для файла ID {file_blocks[0,0]}: {file_blocks[:, 1].tolist()}")

    print("\n--- Визуальная проверка Zero Padding ---")
    # Ищем последний блок любого файла
    # Это блок, за которым следует block_index == 0 или конец массива
    last_block_indices = []
    for i in range(len(L) - 1):
        if L[i+1, 1] == 0:
            last_block_indices.append(i)
    last_block_indices.append(len(L) - 1)

    if last_block_indices:
        idx = last_block_indices[min(2, len(last_block_indices)-1)]
        last_block = X[idx]
        print(f"Последние 16 байт блока {idx} (file_id {L[idx, 0]}):")
        print(last_block[-16:])
        print("(Если в конце нули — Padding работает)")

if __name__ == "__main__":
    verify()
