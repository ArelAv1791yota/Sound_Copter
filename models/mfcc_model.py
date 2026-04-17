# models/mfcc_model.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import glob
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MFCCDataset(Dataset):
    def __init__(self, drone_path, noise_path, progress_callback=None):
        self.features = []
        self.labels = []
        self.progress_callback = progress_callback
        self.last_progress_line = ""

        def log(msg, new_line=True):
            if self.progress_callback:
                try:
                    self.progress_callback(msg, new_line)
                except Exception as e:
                    print(f"Ошибка в log: {e}")

        # Загрузка дронов
        drone_files = glob.glob(os.path.join(drone_path, '*.wav'))
        total_drones = len(drone_files)
        log(f"Загрузка дронов (MFCC): {total_drones} файлов")

        # Первый прогресс с новой строки
        first_progress = True
        for i, file_path in enumerate(drone_files):
            percent = int((i + 1) / total_drones * 100)
            bar_length = 30
            filled = int(bar_length * (i + 1) / total_drones)
            bar = '█' * filled + '░' * (bar_length - filled)
            progress_msg = f"  Дроны: [{bar}] {percent:>3}% ({i + 1}/{total_drones})"

            if first_progress:
                log(progress_msg, new_line=True)  # Первая строка - новая
                first_progress = False
            else:
                log(progress_msg, new_line=False)  # Последующие - перезапись

            features = self._extract_features(file_path)
            if features is not None:
                self.features.append(features)
                self.labels.append(1)

        # После загрузки дронов добавляем пустую строку для разделения
        log("", new_line=True)

        # Загрузка шумов
        noise_files = glob.glob(os.path.join(noise_path, '*.wav'))
        total_noise = len(noise_files)
        log(f"Загрузка шумов (MFCC): {total_noise} файлов")

        first_progress = True
        for i, file_path in enumerate(noise_files):
            percent = int((i + 1) / total_noise * 100)
            bar_length = 30
            filled = int(bar_length * (i + 1) / total_noise)
            bar = '█' * filled + '░' * (bar_length - filled)
            progress_msg = f"  Шумы: [{bar}] {percent:>3}% ({i + 1}/{total_noise})"

            if first_progress:
                log(progress_msg, new_line=True)
                first_progress = False
            else:
                log(progress_msg, new_line=False)

            features = self._extract_features(file_path)
            if features is not None:
                self.features.append(features)
                self.labels.append(0)

        log("", new_line=True)
        log(f"✅ Загружено {len(self.features)} образцов")

    def _extract_features(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=config.MFCC_SR, duration=config.DURATION)
            features = []

            # MFCC (13 коэффициентов)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfcc, axis=1))

            # Спектральный центроид
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            features.append(np.mean(centroid))

            # Спектральная полоса
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
            features.append(np.mean(bandwidth))

            # RMS энергия
            rms = librosa.feature.rms(y=audio)
            features.append(np.mean(rms))

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y=audio)
            features.append(np.mean(zcr))

            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            features.append(np.mean(rolloff))

            return np.array(features, dtype=np.float32)
        except Exception as e:
            print(f"Ошибка {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)


class MFCCDroneDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.MFCC_INPUT_SIZE, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x): return self.network(x)


def extract_features_from_audio(audio, sr):
    """Извлечение признаков из массива аудио"""
    try:
        features = []

        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))

        # Спектральный центроид
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(centroid))

        # Спектральная полоса
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(np.mean(bandwidth))

        # RMS энергия (без sr)
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))

        # Zero crossing rate (без sr)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        features.append(np.mean(zcr))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(rolloff))

        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"Ошибка извлечения признаков: {e}")
        return None


def train_mfcc_model(drone_path, noise_path, params, progress_callback=None, model_name="mfcc_model"):
    def emit(msg):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception as e:
                print(f"Ошибка emit: {e}")
        else:
            print(f"[NO CALLBACK] {msg}")  # Для отладки

    print(f"train_mfcc_model начат, model_name={model_name}")

    if params['batch_size'] < 2:
        emit(f"⚠️ Batch size {params['batch_size']} установлен на 2")
        params['batch_size'] = 2

    emit("Создание датасета MFCC+...")

    # Создаем датасет с callback
    dataset = MFCCDataset(drone_path, noise_path, progress_callback)

    emit(f"Датасет создан, образцов: {len(dataset)}")

    if len(dataset) == 0:
        emit("Ошибка: не найдено аудиофайлов!")
        return None

    indices = list(range(len(dataset)))
    labels = dataset.labels
    train_idx, temp = train_test_split(indices, test_size=params['val_ratio'] + params['test_ratio'], random_state=42,
                                       stratify=labels)
    val_idx, test_idx = train_test_split(temp,
                                         test_size=params['test_ratio'] / (params['val_ratio'] + params['test_ratio']),
                                         random_state=42, stratify=[labels[i] for i in temp])

    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=params['batch_size'],
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=params['batch_size'], shuffle=False)

    if len(train_loader) == 0:
        emit("⚠️ Недостаточно данных для обучения!")
        return None
    emit(f"Train: {len(train_idx)} ({len(train_loader)} батчей), Val: {len(val_idx)}, Test: {len(test_idx)}")

    model = MFCCDroneDetector().to(config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'best_val_acc': 0}

    for epoch in range(params['epochs']):
        # Обучение
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # Валидация
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(config.DEVICE), labels.to(config.DEVICE).unsqueeze(1)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        scheduler.step(avg_val_loss)

        if val_acc > history['best_val_acc']:
            history['best_val_acc'] = val_acc
            model_path = os.path.join(config.MODELS_DIR, f"{model_name}.pth")
            torch.save({'model_state_dict': model.state_dict(), 'input_size': config.MFCC_INPUT_SIZE,
                        'history': history, 'model_name': model_name, 'best_val_acc': val_acc}, model_path)
            emit(f"✅ Модель сохранена: {model_name}.pth (Val Acc: {val_acc:.2f}%)")

        if (epoch + 1) % 10 == 0:
            emit(f"MFCC+ Epoch {epoch + 1}/{params['epochs']}: Val Acc = {val_acc:.2f}%")

    # Тестирование
    if len(test_loader) > 0:
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(config.DEVICE)
                outputs = model(features)
                test_preds.extend((outputs > 0.5).float().cpu().numpy().flatten())
                test_labels.extend(labels.numpy())
        acc = accuracy_score(test_labels, test_preds)
        p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
        history['test_metrics'] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
        emit(f"\n📊 MFCC+ Test: Acc={acc:.3f}, Prec={p:.3f}, Rec={r:.3f}, F1={f1:.3f}")

    # Сохраняем лог (это просто JSON, безопасно)
    config.save_training_log(model_name, "MFCC", history, params)
    emit("MFCC+ обучение завершено")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history