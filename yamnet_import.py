# yamnet_import.py (добавьте эти функции)
import numpy as np
import pickle
import queue
import threading
import subprocess
import os
import sys

_yamnet_process = None
_yamnet_lock = threading.Lock()
_response_queue = queue.Queue()


def _start_yamnet_process():
    """Запуск отдельного процесса для YAMNet"""
    global _yamnet_process

    if _yamnet_process is not None:
        return True

    try:
        script_path = os.path.join(os.path.dirname(__file__), "yamnet_process.py")
        if not os.path.exists(script_path):
            print(f"⚠️ Файл {script_path} не найден")
            return False

        _yamnet_process = subprocess.Popen(
            [sys.executable, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )

        def read_responses():
            while True:
                line = _yamnet_process.stdout.readline()
                if not line:
                    break
                try:
                    data = pickle.loads(bytes.fromhex(line.strip()))
                    _response_queue.put(data)
                except:
                    pass

        threading.Thread(target=read_responses, daemon=True).start()
        print("✅ YAMNet процесс запущен")
        return True
    except Exception as e:
        print(f"❌ Ошибка запуска YAMNet процесса: {e}")
        return False


def _send_request(command, *args):
    """Отправка запроса в YAMNet процесс"""
    if not _start_yamnet_process():
        return None

    msg = "|".join([command] + list(args))
    _yamnet_process.stdin.write(msg + "\n")
    _yamnet_process.stdin.flush()

    try:
        return _response_queue.get(timeout=30)
    except queue.Empty:
        print("⚠️ Таймаут ожидания ответа от YAMNet процесса")
        return None


def get_yamnet_model(model_dir=None):
    """Возвращает None - модель используется через процесс"""
    # Для совместимости возвращаем None, но процесс запускаем
    _start_yamnet_process()
    return None  # Важно: возвращаем None, а не строку!


def get_embedding(file_path):
    """Получение эмбеддинга через отдельный процесс"""
    result = _send_request("embedding", file_path)
    if result is not None:
        return result.astype(np.float32)  # Приводим к float32
    return np.zeros(1024, dtype=np.float32)


def get_embeddings_batch(file_paths):
    """Получение эмбеддингов для нескольких файлов"""
    result = _send_request("batch", *file_paths)
    if result is not None:
        return {path: emb.astype(np.float32) for path, emb in result.items()}
    return {path: np.zeros(1024, dtype=np.float32) for path in file_paths}


def is_available():
    """Проверка доступности"""
    return _start_yamnet_process()