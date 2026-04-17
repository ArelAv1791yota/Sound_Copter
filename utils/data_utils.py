import os
import glob
import numpy as np
import librosa
import config


def load_audio_files(drone_path, noise_path, duration=config.DURATION, sr=config.MFCC_SR):
    """Загрузка аудиофайлов"""
    audio_data = []
    labels = []

    # Загрузка дронов
    drone_files = glob.glob(os.path.join(drone_path, '*.wav'))
    for file_path in drone_files:
        try:
            audio, _ = librosa.load(file_path, sr=sr, duration=duration)
            audio_data.append(audio)
            labels.append(1)
        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")

    # Загрузка шумов
    noise_files = glob.glob(os.path.join(noise_path, '*.wav'))
    for file_path in noise_files:
        try:
            audio, _ = librosa.load(file_path, sr=sr, duration=duration)
            audio_data.append(audio)
            labels.append(0)
        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")

    return np.array(audio_data), np.array(labels)


def prepare_dataset(X, y, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Подготовка датасета с разделением"""
    from sklearn.model_selection import train_test_split

    n = len(X)
    indices = np.arange(n)

    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio),
        random_state=42, stratify=y
    )

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio / (val_ratio + test_ratio),
        random_state=42, stratify=y[temp_idx]
    )

    return train_idx, val_idx, test_idx


def get_file_info(file_path):
    """Получение информации о файле"""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        duration = len(audio) / sr
        return {
            'name': os.path.basename(file_path),
            'path': file_path,
            'duration': duration,
            'sr': sr
        }
    except:
        return None