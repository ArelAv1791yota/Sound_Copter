# models/yamnet_model.py
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
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from yamnet_import import get_yamnet_model


class YAMNetDataset(Dataset):
    def __init__(self, drone_path, noise_path, cache_dir='yamnet_cache', progress_callback=None):
        self.cache_dir = os.path.join(config.BASE_DIR, cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.yamnet = get_yamnet_model(config.YAMNET_MODEL_DIR)
        self.embeddings, self.labels = [], []
        self.progress_callback = progress_callback

        def log(msg, new_line=True):
            if self.progress_callback:
                try:
                    self.progress_callback(msg, new_line)
                except:
                    pass

        # Загрузка дронов
        drone_files = glob.glob(os.path.join(drone_path, '*.wav'))
        total_drones = len(drone_files)
        log(f"Загрузка дронов (YAMNet): {total_drones} файлов")

        first_progress = True
        for i, file_path in enumerate(drone_files):
            percent = int((i + 1) / total_drones * 100)
            bar_length = 30
            filled = int(bar_length * (i + 1) / total_drones)
            bar = '█' * filled + '░' * (bar_length - filled)
            progress_msg = f"  Дроны: [{bar}] {percent:>3}% ({i + 1}/{total_drones})"

            if first_progress:
                log(progress_msg, new_line=True)
                first_progress = False
            else:
                log(progress_msg, new_line=False)

            emb = self._get_embedding(file_path)
            if emb is not None:
                self.embeddings.append(emb)
                self.labels.append(1)

        log("", new_line=True)

        # Загрузка шумов
        noise_files = glob.glob(os.path.join(noise_path, '*.wav'))
        total_noise = len(noise_files)
        log(f"Загрузка шумов (YAMNet): {total_noise} файлов")

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

            emb = self._get_embedding(file_path)
            if emb is not None:
                self.embeddings.append(emb)
                self.labels.append(0)

        log("", new_line=True)
        log(f"✅ Загружено {len(self.embeddings)} образцов")

    def _get_embedding(self, file_path):
        cache_file = os.path.join(self.cache_dir, os.path.basename(file_path) + '.npy')
        if os.path.exists(cache_file): return np.load(cache_file)
        try:
            audio, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, mono=True)
            chunk_size, embeddings = 15600, []
            for i in range(0, len(audio), chunk_size // 2):
                chunk = audio[i:i + chunk_size]
                if len(chunk) < chunk_size: chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                chunk = np.clip(chunk, -1, 1).astype(np.float32)
                if self.yamnet is None: return None
                scores, emb_chunk, _ = self.yamnet(chunk)
                embeddings.append(emb_chunk.numpy().mean(axis=0))
            if embeddings:
                emb = np.mean(embeddings, axis=0)
                np.save(cache_file, emb)
                return emb
            return np.zeros(config.YAMNET_EMBEDDING_SIZE, dtype=np.float32)
        except Exception as e:
            print(f"Ошибка {file_path}: {e}")
            return None

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx]), torch.tensor(self.labels[idx], dtype=torch.float32)


class YAMNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(config.YAMNET_EMBEDDING_SIZE, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, x): return self.fc(x)


def get_yamnet_embedding(audio):
    yamnet = config.get_yamnet_model()
    if yamnet is None: return np.zeros(config.YAMNET_EMBEDDING_SIZE, dtype=np.float32)
    chunk_size, embeddings = 15600, []
    for i in range(0, len(audio), chunk_size // 2):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size: chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunk = np.clip(chunk, -1, 1).astype(np.float32)
        try:
            scores, emb_chunk, _ = yamnet(chunk)
            embeddings.append(emb_chunk.numpy().mean(axis=0))
        except:
            continue
    if embeddings: return np.mean(embeddings, axis=0).astype(np.float32)
    return np.zeros(config.YAMNET_EMBEDDING_SIZE, dtype=np.float32)


def train_yamnet_model(drone_path, noise_path, params, progress_callback=None, model_name="yamnet_model"):
    def emit(msg):
        progress_callback(msg) if progress_callback else None

    if params['batch_size'] < 2:
        emit(f"⚠️ Batch size {params['batch_size']} установлен на 2")
        params['batch_size'] = 2

    emit("Создание датасета YAMNet...")
    drone_files, noise_files = glob.glob(os.path.join(drone_path, '*.wav')), glob.glob(
        os.path.join(noise_path, '*.wav'))
    emit(f"Найдено дронов: {len(drone_files)}, шумов: {len(noise_files)}")
    if not drone_files or not noise_files: emit("Ошибка: не найдены аудиофайлы!"); return None

    dataset = YAMNetDataset(drone_path, noise_path, progress_callback=progress_callback)  # Передаем callback

    if len(dataset) == 0: emit("Ошибка: не удалось загрузить данные!"); return None

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

    if len(train_loader) == 0: emit("⚠️ Недостаточно данных для обучения!"); return None
    emit(f"Train: {len(train_idx)} ({len(train_loader)} батчей), Val: {len(val_idx)}, Test: {len(test_idx)}")

    model = YAMNetClassifier().to(config.DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'best_val_acc': 0}

    for epoch in range(params['epochs']):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for emb, labels in train_loader:
            emb, labels = emb.to(config.DEVICE), labels.to(config.DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(emb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for emb, labels in val_loader:
                emb, labels = emb.to(config.DEVICE), labels.to(config.DEVICE).unsqueeze(1)
                outputs = model(emb)
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
            torch.save({'model_state_dict': model.state_dict(), 'input_dim': config.YAMNET_EMBEDDING_SIZE,
                        'history': history, 'model_name': model_name, 'best_val_acc': val_acc}, model_path)
            emit(f"✅ Модель сохранена: {model_name}.pth (Val Acc: {val_acc:.2f}%)")

        if (epoch + 1) % 10 == 0:
            emit(f"YAMNet Epoch {epoch + 1}/{params['epochs']}: Val Acc = {val_acc:.2f}%")

    # Тестирование
    if len(test_loader) > 0:
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for emb, labels in test_loader:
                emb = emb.to(config.DEVICE)
                outputs = model(emb)
                test_preds.extend((outputs > 0.5).float().cpu().numpy().flatten())
                test_labels.extend(labels.numpy())
        acc = accuracy_score(test_labels, test_preds)
        p, r, f1, _ = precision_recall_fscore_support(test_labels, test_preds, average='binary')
        history['test_metrics'] = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
        emit(f"\n📊 YAMNet Test: Acc={acc:.3f}, Prec={p:.3f}, Rec={r:.3f}, F1={f1:.3f}")

    # # Сохраняем график
    # try:
    #     from utils.plot_utils import plot_training_history
    #     plot_path = os.path.join(config.PLOTS_DIR, f"{model_name}_training_history.png")
    #     plot_training_history(history, title=model_name, save_path=plot_path)
    #     emit(f"📊 График сохранен: {plot_path}")
    # except Exception as e:
    #     print(f"Ошибка сохранения графика: {e}")

    config.save_training_log(model_name, "YAMNet", history, params)
    emit("YAMNet обучение завершено")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return history