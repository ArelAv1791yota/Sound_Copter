# config.py
import torch
import os
import sys
import json
from datetime import datetime

# Пути
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = ROOT_DIR
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
PLOTS_DIR = os.path.join(MODELS_DIR, "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "training_logs")
YAMNET_MODEL_DIR = os.path.join(BASE_DIR, "models", "yamnet")

# Создаем папки
for dir_path in [MODELS_DIR, PLOTS_DIR, RESULTS_DIR, LOGS_DIR, YAMNET_MODEL_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Общие параметры
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_RATE = 16000
DURATION = 1.0

# Параметры MFCC модели
MFCC_INPUT_SIZE = 18
MFCC_SR = 22050

# Параметры YAMNet модели
YAMNET_EMBEDDING_SIZE = 1024

# Параметры обучения
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_TRAIN_RATIO = 70
DEFAULT_VAL_RATIO = 15
DEFAULT_TEST_RATIO = 15


def save_training_log(model_name, model_type, history, params):
    """Сохранение лога обучения"""
    try:
        log_data = {
            'model_name': model_name,
            'model_type': model_type,
            'timestamp': datetime.now().isoformat(),
            'params': params,
            'history': history,
            'best_val_acc': history.get('best_val_acc', 0),
            'test_metrics': history.get('test_metrics', {})
        }
        safe_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(LOGS_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        return filepath
    except Exception as e:
        print(f"Ошибка сохранения лога: {e}")
        return None


def load_training_logs():
    """Загрузка всех логов обучения"""
    logs = []
    for file in os.listdir(LOGS_DIR):
        if file.endswith('.json'):
            with open(os.path.join(LOGS_DIR, file), 'r', encoding='utf-8') as f:
                logs.append(json.load(f))
    return logs


def get_available_models():
    """Получение списка доступных моделей с их логами"""
    models = []
    for file in os.listdir(MODELS_DIR):
        if file.endswith('.pth'):
            model_path = os.path.join(MODELS_DIR, file)
            model_name = file[:-4]
            log_files = [f for f in os.listdir(LOGS_DIR) if f.startswith(model_name)]
            if log_files:
                latest_log = sorted(log_files)[-1]
                with open(os.path.join(LOGS_DIR, latest_log), 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                models.append({'model_name': model_name, 'model_path': model_path, 'log': log_data})
            else:
                models.append({'model_name': model_name, 'model_path': model_path, 'log': None})
    return models


# YAMNet функции (ленивая загрузка)
def get_yamnet_model():
    try:
        from yamnet_import import get_yamnet_model as _get_model
        return _get_model()
    except Exception as e:
        print(f"YAMNet недоступен: {e}")
        return None


def check_yamnet_availability():
    try:
        from yamnet_import import is_available
        return is_available()
    except:
        return False


YAMNET_AVAILABLE = check_yamnet_availability()
print(f"YAMNet доступность: {YAMNET_AVAILABLE}")
print(f"ROOT_DIR: {ROOT_DIR}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"LOGS_DIR: {LOGS_DIR}")
print(f"YAMNET_MODEL_DIR: {YAMNET_MODEL_DIR}")
print(f"YAMNET_AVAILABLE: {YAMNET_AVAILABLE}")