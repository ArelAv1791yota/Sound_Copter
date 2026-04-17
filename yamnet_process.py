# yamnet_process.py
import sys
import os
import pickle
import numpy as np
import time

# Устанавливаем переменные окружения ДО импорта
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Импортируем TensorFlow
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# Подавляем предупреждения
tf.get_logger().setLevel('ERROR')

print("YAMNet процесс запущен, загрузка модели...", file=sys.stderr)

# Загружаем модель
try:
    model_dir = os.path.join(os.path.dirname(__file__), "models", "yamnet")
    if os.path.exists(os.path.join(model_dir, "saved_model.pb")):
        yamnet = hub.load(model_dir)
        print("Локальная YAMNet загружена", file=sys.stderr)
    else:
        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        print("YAMNet загружена из интернета", file=sys.stderr)
except Exception as e:
    print(f"Ошибка загрузки YAMNet: {e}", file=sys.stderr)
    yamnet = None


def get_embedding(file_path):
    """Получение эмбеддинга для файла"""
    if yamnet is None:
        return np.zeros(1024, dtype=np.float32)

    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        chunk_size = 15600
        embeddings = []

        for i in range(0, len(audio), chunk_size // 2):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            chunk = np.clip(chunk, -1, 1).astype(np.float32)

            scores, embeddings_chunk, spectrogram = yamnet(chunk)
            # Берем среднее по времени для этого чанка
            chunk_embedding = embeddings_chunk.numpy().mean(axis=0)
            embeddings.append(chunk_embedding)

        if embeddings:
            result = np.mean(embeddings, axis=0).astype(np.float32)
            return result
        return np.zeros(1024, dtype=np.float32)
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}", file=sys.stderr)
        return np.zeros(1024, dtype=np.float32)


def get_embeddings_batch(file_paths):
    """Получение эмбеддингов для списка файлов"""
    results = {}
    for path in file_paths:
        results[path] = get_embedding(path)
    return results


# Основной цикл
print("YAMNet процесс готов к работе", file=sys.stderr)
sys.stderr.flush()

while True:
    try:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line == "exit":
            break

        parts = line.split("|")
        command = parts[0]

        if command == "embedding" and len(parts) >= 2:
            path = parts[1]
            result = get_embedding(path)
            # Отправляем результат
            sys.stdout.write(pickle.dumps(result).hex() + "\n")
            sys.stdout.flush()

        elif command == "batch" and len(parts) >= 2:
            paths = parts[1:]
            results = get_embeddings_batch(paths)
            sys.stdout.write(pickle.dumps(results).hex() + "\n")
            sys.stdout.flush()

    except Exception as e:
        sys.stderr.write(f"Ошибка в цикле: {e}\n")
        sys.stderr.flush()