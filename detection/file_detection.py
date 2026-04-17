# detection/file_detection.py
import torch
import librosa
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.mfcc_model import MFCCDroneDetector, extract_features_from_audio
from models.yamnet_model import YAMNetClassifier


class FileDetector:
    def __init__(self):
        self.device = config.DEVICE
        self.mfcc_model = None
        self.yamnet_model = None

    def load_mfcc_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.mfcc_model = MFCCDroneDetector().to(self.device)
            self.mfcc_model.load_state_dict(checkpoint['model_state_dict'])
            self.mfcc_model.eval()
            print(f"✅ MFCC модель загружена: {os.path.basename(model_path)}")
            return True
        except Exception as e:
            print(f"Ошибка загрузки MFCC модели: {e}")
            return False

    def load_yamnet_model(self, model_path):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.yamnet_model = YAMNetClassifier().to(self.device)
            self.yamnet_model.load_state_dict(checkpoint['model_state_dict'])
            self.yamnet_model.eval()
            print(f"✅ YAMNet модель загружена: {os.path.basename(model_path)}")
            return True
        except Exception as e:
            print(f"Ошибка загрузки YAMNet модели: {e}")
            return False

    def detect_file(self, file_path):
        result = {'file': file_path, 'duration': 0, 'mfcc_prob': None, 'mfcc_result': 'N/A',
                  'yamnet_prob': None, 'yamnet_result': 'N/A'}

        try:
            audio, sr = librosa.load(file_path, sr=None)
            result['duration'] = len(audio) / sr
        except:
            pass

        if self.mfcc_model:
            try:
                audio, sr = librosa.load(file_path, sr=config.MFCC_SR, duration=config.DURATION)
                features = extract_features_from_audio(audio, sr)
                if features is not None:
                    prob = self.mfcc_model(
                        torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)).item()
                    result['mfcc_prob'] = prob
                    result['mfcc_result'] = "ДА (дрон)" if prob > 0.5 else "НЕТ"
            except Exception as e:
                print(f"Ошибка MFCC: {e}")

        if self.yamnet_model:
            try:
                from yamnet_import import get_embedding
                embedding = get_embedding(file_path)
                prob = self.yamnet_model(
                    torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(self.device)).item()
                result['yamnet_prob'] = prob
                result['yamnet_result'] = "ДА (дрон)" if prob > 0.5 else "НЕТ"
            except Exception as e:
                print(f"Ошибка YAMNet: {e}")

        return result