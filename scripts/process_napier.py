import os
import zipfile
import shutil
import tempfile
import json
import numpy as np
from tqdm import tqdm
import array
import subprocess



def extract_nested_zips(base_dir):
    """
    Рекурсивно ищет и распаковывает все архивы с помощью 7z.
    Это исправляет ошибки с PPMD, LZMA и шифрованием.
    """
    while True:
        zips_found = False
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                lower_file = file.lower()
                # 7z поддерживает почти все: zip, rar, 7z, tar, gz и т.д.
                if lower_file.endswith('.zip') or lower_file.endswith('.rar') or lower_file.endswith('.7z'):
                    zips_found = True
                    zip_path = os.path.join(root, file)
                    extract_dir = os.path.splitext(zip_path)[0] + "_extracted"
                    os.makedirs(extract_dir, exist_ok=True)
                    
                    try:
                        # Используем 7z для распаковки. 
                        # -y: отвечать "да" на всё
                        # -o: папка вывода
                        # -p"": пустой пароль (для некоторых архивов Napier)
                        subprocess.run(
                            ['7z', 'x', zip_path, f'-o{extract_dir}', '-y', '-p""'],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            check=False
                        )
                    except Exception as e:
                        print(f"\n[Warning] Ошибка 7z при распаковке {zip_path}: {e}")
                    
                    try:
                        os.remove(zip_path)
                    except OSError:
                        pass
                        
        if not zips_found:
            break

def main():
    # Настройки
    data_dir = "/home/arsen/file-carver/data/reassembly/NapierOne-tiny"
    output_x = "napier_X.npy"
    output_y = "napier_Y.npy"
    output_l = "napier_L.npy"
    output_json = "classes.json"
    block_size = 512

    if not os.path.exists(data_dir):
        print(f"Ошибка: Директория {data_dir} не существует.")
        return

    # 1. Определяем источники данных (папки или архивы)
    all_items = sorted(os.listdir(data_dir))
    class_sources = {} 

    for item in all_items:
        full_path = os.path.join(data_dir, item)
        if os.path.isdir(full_path):
            class_sources[item] = full_path
        elif item.lower().endswith(('.zip', '.rar', '.7z')):
            # Извлекаем имя класса
            cls_name = item.replace("-tiny.zip", "").replace(".zip", "").replace(".rar", "").replace(".7z", "")
            class_sources[cls_name] = full_path

    classes = sorted(list(class_sources.keys()))
    class_to_id = {cls_name: i for i, cls_name in enumerate(classes)}

    if not classes:
        print(f"Ошибка: В {data_dir} не найдено ни папок, ни архивов.")
        return

    # Сохраняем маппинг классов в JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(class_to_id, f, indent=4)
    print(f"Сохранен файл {output_json} (всего классов: {len(classes)}).")

    # Инициализируем временные файлы для хранения данных на диске
    temp_x_path = "temp_x.bin"
    temp_y_class_path = "temp_y_class.bin"
    temp_y_link_path = "temp_y_link.bin"

    file_id_counter = 0
    total_blocks = 0

    try:
        with open(temp_x_path, "wb") as fx, \
             open(temp_y_class_path, "wb") as fy_c, \
             open(temp_y_link_path, "wb") as fy_l:

            for cls_name in tqdm(classes, desc="Обработка классов"):
                cls_id = class_to_id[cls_name]
                source_path = class_sources[cls_name]
                
                with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
                    temp_cls_dir = os.path.join(temp_dir, "work_dir")
                    os.makedirs(temp_cls_dir, exist_ok=True)

                    if os.path.isdir(source_path):
                        shutil.copytree(source_path, temp_cls_dir, dirs_exist_ok=True)
                    else:
                        try:
                            # Используем 7z для корневого архива
                            subprocess.run(
                                ['7z', 'x', source_path, f'-o{temp_cls_dir}', '-y', '-p""'],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                check=False
                            )
                        except Exception as e:
                            print(f"\n[Error] Не удалось распаковать корневой архив {source_path}: {e}")
                            continue
                    
                    extract_nested_zips(temp_cls_dir)
                    
                    all_files = []
                    for root, dirs, files in os.walk(temp_cls_dir):
                        for file in files:
                            all_files.append(os.path.join(root, file))
                            
                    for file_path in tqdm(all_files, desc=f"Нарезка: {cls_name}", leave=False):
                        try:
                            file_size = os.path.getsize(file_path)
                            if file_size == 0:
                                continue
                                
                            num_blocks = (file_size + block_size - 1) // block_size
                            
                            with open(file_path, "rb") as f:
                                for block_idx in range(num_blocks):
                                    block = f.read(block_size)
                                    
                                    # Zero padding
                                    if len(block) < block_size:
                                        block = block.ljust(block_size, b'\x00')
                                        
                                    fx.write(block)
                                    
                                    # Записываем метки в бинарном виде
                                    fy_c.write(np.array([cls_id], dtype=np.uint16).tobytes())
                                    fy_l.write(np.array([file_id_counter, block_idx], dtype=np.uint32).tobytes())
                                    
                                    total_blocks += 1
                                    
                            file_id_counter += 1
                        except Exception as e:
                            print(f"\n[Error] Ошибка обработки файла {file_path}: {e}")

        print("\nКонвертация данных в финальные массивы .npy...")
        
        # 4. Сохраняем как отдельные .npy файлы
        X_mem = np.memmap(temp_x_path, dtype=np.uint8, mode='r', shape=(total_blocks, block_size))
        np.save(output_x, X_mem)
        
        y_class_dtype = np.uint8 if len(classes) < 256 else np.uint16
        Y_class_data = np.fromfile(temp_y_class_path, dtype=np.uint16).astype(y_class_dtype)
        np.save(output_y, Y_class_data)
        
        Y_link_mem = np.memmap(temp_y_link_path, dtype=np.uint32, mode='r', shape=(total_blocks, 2))
        np.save(output_l, Y_link_mem)

        print(f"Всего обработано файлов: {file_id_counter}")
        print(f"Всего блоков (512 байт): {total_blocks}")
        print(f"Файлы сохранены: {output_x}, {output_y}, {output_l}")
        print("Готово! Пайплайн успешно завершен.")

    finally:
        # Удаляем временные файлы
        for p in [temp_x_path, temp_y_class_path, temp_y_link_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass

if __name__ == "__main__":
    main()
