# detection/micro_detection.py
import pyaudio
import numpy as np
import torch
import librosa
import threading
import queue
import time
import os
import sys
from PyQt5.QtCore import QObject, pyqtSignal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from detection.file_detection import FileDetector
from models.mfcc_model import extract_features_from_audio  # ИСПРАВЛЕНО


class MicrophoneDetector(QObject):
    """Детекция дрона с микрофона в реальном времени"""

    result_ready = pyqtSignal(dict)  # Сигнал с результатом детекции
    level_update = pyqtSignal(int)  # Сигнал уровня звука
    spectrum_ready = pyqtSignal(np.ndarray, np.ndarray)  # Сигнал спектра
    drone_frequency_ready = pyqtSignal(list)  # Сигнал с частотами дронов

    def __init__(self):
        super().__init__()
        self.detector = FileDetector()
        self.is_recording = False
        self.audio_queue = queue.Queue()

        # Параметры записи
        self.format = pyaudio.paFloat32
        self.channels = 1
        self.rate = config.SAMPLE_RATE  # 16000 Гц для YAMNet
        self.chunk = 1024
        self.update_interval = 0.5  # Обновление каждые 0.5 секунды

        # Для накопления аудио
        self.audio_buffer = np.array([], dtype=np.float32)

        # Частоты, характерные для дронов (можно настроить)
        self.drone_frequencies = [150, 300, 600, 1200, 2400, 4800]  # Гц
        self.drone_freq_range = (100, 5000)  # Диапазон частот дронов

    def start_detection(self, use_mfcc=True, use_yamnet=True):
        """Запуск детекции в реальном времени"""
        self.is_recording = True
        self.use_mfcc = use_mfcc
        self.use_yamnet = use_yamnet
        self.audio_buffer = np.array([], dtype=np.float32)

        # Запускаем поток записи
        self.record_thread = threading.Thread(target=self.record_audio)
        self.record_thread.daemon = True
        self.record_thread.start()

        # Запускаем поток обработки
        self.process_thread = threading.Thread(target=self.process_audio)
        self.process_thread.daemon = True
        self.process_thread.start()

    def stop(self):
        """Остановка детекции"""
        self.is_recording = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join(timeout=1)
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1)

    def record_audio(self):
        """Непрерывная запись аудио с микрофона"""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        while self.is_recording:
            try:
                data = stream.read(self.chunk, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                # Добавляем в буфер
                self.audio_buffer = np.append(self.audio_buffer, audio_chunk)

                # Обновляем уровень звука
                level = int(np.abs(audio_chunk).mean() * 100)
                self.level_update.emit(min(level, 100))

                # Если накопилось достаточно данных, отправляем на обработку
                samples_needed = int(self.rate * self.update_interval)
                if len(self.audio_buffer) >= samples_needed:
                    # Берем последние samples_needed семплов
                    audio_segment = self.audio_buffer[-samples_needed:].copy()
                    self.audio_queue.put(audio_segment)

                    # Очищаем буфер, но оставляем перекрытие для плавности
                    overlap = samples_needed // 2
                    self.audio_buffer = self.audio_buffer[-overlap:] if len(self.audio_buffer) > overlap else np.array([])

            except Exception as e:
                print(f"Ошибка записи: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()

    def process_audio(self):
        """Обработка аудио в реальном времени"""
        import soundfile as sf
        import tempfile

        while self.is_recording:
            try:
                audio = self.audio_queue.get(timeout=0.1)

                # Вычисляем спектр для визуализации
                freq = np.fft.rfftfreq(len(audio), 1 / self.rate)
                spectrum = np.abs(np.fft.rfft(audio))
                max_freq_idx = np.searchsorted(freq, 8000)
                self.spectrum_ready.emit(freq[:max_freq_idx], spectrum[:max_freq_idx])

                # Сохраняем временный файл для детекции
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    tmp_path = tmp.name
                sf.write(tmp_path, audio, self.rate)

                # Детекция через модели (уже с загруженными моделями)
                result = self.detector.detect_file(tmp_path)
                self.result_ready.emit(result)

                # Удаляем временный файл
                try:
                    os.unlink(tmp_path)
                except:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка в процессе обработки: {e}")

    def detect_drone_frequencies(self, freq, spectrum):
        """Обнаружение частот, характерных для дронов"""
        # Находим пики в спектре
        from scipy.signal import find_peaks

        # Нормализуем спектр
        if np.max(spectrum) > 0:
            spectrum_norm = spectrum / np.max(spectrum)
        else:
            return []

        # Ищем пики
        peaks, properties = find_peaks(spectrum_norm, height=0.1, distance=10)

        # Фильтруем пики в диапазоне частот дронов
        drone_peaks = []
        for peak in peaks:
            if self.drone_freq_range[0] <= freq[peak] <= self.drone_freq_range[1]:
                if spectrum_norm[peak] > 0.2:  # Минимальная высота пика
                    drone_peaks.append({
                        'freq': float(freq[peak]),
                        'amplitude': float(spectrum_norm[peak])
                    })

        # Сортируем по амплитуде
        drone_peaks.sort(key=lambda x: x['amplitude'], reverse=True)

        # Возвращаем до 5 самых сильных пиков
        return drone_peaks[:5]