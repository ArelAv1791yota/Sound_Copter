# gui.py
import sys
import os
import threading
import glob
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Импортируем matplotlib ТОЛЬКО один раз с правильным backend
import matplotlib
matplotlib.use('Qt5Agg')  # Только один раз, в начале
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from models.mfcc_model import train_mfcc_model
from models.yamnet_model import train_yamnet_model
from detection.file_detection import FileDetector
from detection.micro_detection import MicrophoneDetector
from utils.plot_utils import plot_training_history

# Настройки matplotlib (только после импорта)
plt.rcParams['figure.facecolor'] = '#1e1e1e'
plt.rcParams['axes.facecolor'] = '#2b2b2b'
plt.rcParams['savefig.facecolor'] = '#1e1e1e'
plt.rcParams['axes.labelcolor'] = '#cccccc'
plt.rcParams['xtick.color'] = '#cccccc'
plt.rcParams['ytick.color'] = '#cccccc'
plt.rcParams['grid.color'] = '#3a3a3a'
plt.style.use('dark_background')


class TrainingThread(QThread):
    progress_update = pyqtSignal(str, bool)  # (message, new_line)
    training_finished = pyqtSignal(dict, dict)

    def __init__(self, drone_path, noise_path, params, train_mfcc, train_yamnet, mfcc_name, yamnet_name):
        super().__init__()
        self.drone_path = drone_path
        self.noise_path = noise_path
        self.params = params
        self.train_mfcc = train_mfcc
        self.train_yamnet = train_yamnet
        self.mfcc_name = mfcc_name  # Всегда сохраняем, даже если не обучаем
        self.yamnet_name = yamnet_name  # Всегда сохраняем, даже если не обучаем

    def run(self):
        try:
            mfcc_history = None
            yamnet_history = None

            def emit_callback(msg, new_line=True):
                self.progress_update.emit(msg, new_line)

            if self.train_mfcc:
                emit_callback(f"Начало обучения MFCC+ модели ({self.mfcc_name})...")
                mfcc_history = train_mfcc_model(
                    self.drone_path,
                    self.noise_path,
                    self.params,
                    emit_callback,
                    model_name=self.mfcc_name
                )

            if self.train_yamnet:
                emit_callback(f"\nНачало обучения YAMNet модели ({self.yamnet_name})...")
                yamnet_history = train_yamnet_model(
                    self.drone_path,
                    self.noise_path,
                    self.params,
                    emit_callback,
                    model_name=self.yamnet_name
                )

            self.training_finished.emit(mfcc_history, yamnet_history)

        except Exception as e:
            self.progress_update.emit(f"Ошибка: {str(e)}", True)
            self.training_finished.emit(None, None)


class DetectionThread(QThread):
    """Поток для детекции"""
    result_ready = pyqtSignal(list)
    progress_update = pyqtSignal(str)

    def __init__(self, files, mfcc_model_path, yamnet_model_path):
        super().__init__()
        self.files = files
        self.mfcc_model_path = mfcc_model_path
        self.yamnet_model_path = yamnet_model_path

    def run(self):
        results = []
        detector = FileDetector()

        if self.mfcc_model_path:
            detector.load_mfcc_model(self.mfcc_model_path)
        if self.yamnet_model_path:
            detector.load_yamnet_model(self.yamnet_model_path)

        for i, file in enumerate(self.files):
            self.progress_update.emit(f"Обработка {i + 1}/{len(self.files)}: {os.path.basename(file)}")
            result = detector.detect_file(file)
            results.append(result)

        self.result_ready.emit(results)


class MicroDetectionThread(QThread):
    """Поток для детекции с микрофона"""
    result_ready = pyqtSignal(dict)
    level_update = pyqtSignal(int)
    spectrum_ready = pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self, mfcc_model_path, yamnet_model_path):
        super().__init__()
        self.mfcc_model_path = mfcc_model_path
        self.yamnet_model_path = yamnet_model_path
        self.is_running = True
        self.detector = FileDetector()

    def run(self):
        if self.mfcc_model_path:
            self.detector.load_mfcc_model(self.mfcc_model_path)
        if self.yamnet_model_path:
            self.detector.load_yamnet_model(self.yamnet_model_path)

        import pyaudio
        import soundfile as sf
        import tempfile

        # Параметры записи
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = config.SAMPLE_RATE
        CHUNK = 1024
        UPDATE_INTERVAL = 0.5  # секунд
        samples_per_update = int(RATE * UPDATE_INTERVAL)

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)

        audio_buffer = np.array([], dtype=np.float32)

        while self.is_running:
            try:
                # Читаем данные
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                audio_buffer = np.append(audio_buffer, audio_chunk)

                # Уровень звука
                level = int(np.abs(audio_chunk).mean() * 100)
                self.level_update.emit(min(level, 100))

                # Если накопилось достаточно для обработки
                if len(audio_buffer) >= samples_per_update:
                    # Берем последние samples_per_update семплов
                    audio_segment = audio_buffer[-samples_per_update:].copy()

                    # Спектр
                    freq = np.fft.rfftfreq(len(audio_segment), 1 / RATE)
                    spectrum = np.abs(np.fft.rfft(audio_segment))
                    self.spectrum_ready.emit(freq[:len(freq) // 2], spectrum[:len(spectrum) // 2])

                    # Сохраняем во временный файл
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                        tmp_path = tmp.name
                    sf.write(tmp_path, audio_segment, RATE)

                    # Детекция
                    result = self.detector.detect_file(tmp_path)
                    self.result_ready.emit(result)

                    # Удаляем временный файл
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

                    # Оставляем перекрытие
                    overlap = samples_per_update // 2
                    audio_buffer = audio_buffer[-overlap:] if len(audio_buffer) > overlap else np.array([])

            except Exception as e:
                print(f"Ошибка в микрофоне: {e}")

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self.is_running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mfcc_models = []
        self.yamnet_models = []
        self.micro_thread = None
        self.plot_window = None
        self.mfcc_history = None
        self.yamnet_history = None
        self.last_results = None

        self.initUI()
        self.load_available_models()

    def initUI(self):
        self.setWindowTitle("Sound_Copter - Детекция дронов")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        training_tab = self.create_training_tab()
        tabs.addTab(training_tab, "Обучение моделей")

        file_tab = self.create_file_detection_tab()
        tabs.addTab(file_tab, "Детекция файлов")

        micro_tab = self.create_microphone_tab()
        tabs.addTab(micro_tab, "Детекция с микрофона")

        plots_tab = self.create_plots_tab()
        tabs.addTab(plots_tab, "Графики обучения")

        # Устанавливаем темный стиль для matplotlib
        plt.style.use('dark_background')

        # Устанавливаем параметры matplotlib глобально
        plt.rcParams['figure.facecolor'] = '#1e1e1e'
        plt.rcParams['axes.facecolor'] = '#2b2b2b'
        plt.rcParams['savefig.facecolor'] = '#1e1e1e'
        plt.rcParams['figure.edgecolor'] = '#1e1e1e'
        plt.rcParams['axes.edgecolor'] = '#3a3a3a'
        plt.rcParams['axes.labelcolor'] = '#cccccc'
        plt.rcParams['xtick.color'] = '#cccccc'
        plt.rcParams['ytick.color'] = '#cccccc'
        plt.rcParams['text.color'] = '#cccccc'
        plt.rcParams['grid.color'] = '#3a3a3a'

        # Настройка цветов для графиков
        plt.rcParams['figure.facecolor'] = '#1e1e1e'
        plt.rcParams['axes.facecolor'] = '#2b2b2b'
        plt.rcParams['savefig.facecolor'] = '#1e1e1e'

        self.statusBar().showMessage("Готов к работе")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QTabWidget::pane {
                border: 1px solid #3a3a3a;
                background-color: #252526;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QTabBar::tab:hover {
                background-color: #3a3a3a;
            }
            QLabel {
                color: #cccccc;
                font-size: 12px;
            }
            QPushButton {
                background-color: #3c3c3c;
                color: #ffffff;
                border: 1px solid #5a5a5a;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #6a6a6a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #6a6a6a;
                border: 1px solid #3a3a3a;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New';
                border: 1px solid #3a3a3a;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3a3a3a;
                padding: 4px;
                border-radius: 3px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #6a6a6a;
            }
            QGroupBox {
                color: #cccccc;
                border: 1px solid #3a3a3a;
                margin-top: 1ex;
                border-radius: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #cccccc;
                margin-right: 5px;
            }
            QScrollBar:vertical {
                background-color: #2d2d2d;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #5a5a5a;
                border-radius: 6px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6a6a6a;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QTableWidget {
                background-color: #252526;
                color: #cccccc;
                gridline-color: #3a3a3a;
                border: 1px solid #3a3a3a;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #3c3c3c;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 4px;
                border: 1px solid #3a3a3a;
            }
            QProgressBar {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
                border-radius: 3px;
                text-align: center;
                color: #cccccc;
            }
            QProgressBar::chunk {
                background-color: #4a90e2;
                border-radius: 3px;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #cccccc;
            }
            QMenuBar::item:selected {
                background-color: #3c3c3c;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #cccccc;
                border: 1px solid #3a3a3a;
            }
            QMenu::item:selected {
                background-color: #3c3c3c;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid #5a5a5a;
                background-color: #2d2d2d;
            }
            QCheckBox::indicator:checked {
                background-color: #4a90e2;
                border: 1px solid #4a90e2;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #6a6a6a;
            }
            QTableWidget {
                background-color: #252526;
                color: #cccccc;
                gridline-color: #3a3a3a;
                border: 1px solid #3a3a3a;
                alternate-background-color: #2d2d2d;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #3c3c3c;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 5px;
                border: 1px solid #3a3a3a;
            }
            QTableCornerButton::section {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
            }
            
            QGroupBox {
                background-color: #252526;
            }
            
            QScrollArea {
                background-color: #1e1e1e;
                border: none;
            }
            
            QTabWidget::pane {
                background-color: #252526;
            }
        """)

    def create_plots_tab(self):
        """Вкладка с графиками обучения"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Верхняя часть - выбор моделей
        select_group = QGroupBox("Выбор моделей для сравнения")
        select_layout = QGridLayout()

        select_layout.addWidget(QLabel("Модель 1:"), 0, 0)
        self.plot_model1_combo = QComboBox()
        select_layout.addWidget(self.plot_model1_combo, 0, 1)

        select_layout.addWidget(QLabel("Модель 2:"), 1, 0)
        self.plot_model2_combo = QComboBox()
        select_layout.addWidget(self.plot_model2_combo, 1, 1)

        self.load_plots_btn = QPushButton("Загрузить графики")
        self.load_plots_btn.clicked.connect(self.load_plots)
        select_layout.addWidget(self.load_plots_btn, 2, 0, 1, 2)

        select_group.setLayout(select_layout)
        layout.addWidget(select_group)

        # Графики на всю ширину - создаем фигуру с темным фоном
        self.plot_figure = plt.figure(figsize=(12, 5), facecolor='#1e1e1e')
        self.plot_canvas = FigureCanvas(self.plot_figure)
        layout.addWidget(self.plot_canvas)

        # Устанавливаем начальное сообщение
        self.plot_figure.clear()
        ax1 = self.plot_figure.add_subplot(121)
        ax2 = self.plot_figure.add_subplot(122)

        ax1.set_facecolor('#2b2b2b')
        ax2.set_facecolor('#2b2b2b')

        ax1.text(0.5, 0.5, 'Выберите модели и нажмите "Загрузить графики"',
                 ha='center', va='center', fontsize=10, color='#cccccc')
        ax2.text(0.5, 0.5, 'Выберите модели и нажмите "Загрузить графики"',
                 ha='center', va='center', fontsize=10, color='#cccccc')

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

        self.plot_canvas.draw()

        # Объединенная таблица метрик
        metrics_group = QGroupBox("Метрики моделей")
        metrics_group.setStyleSheet("""
            QGroupBox {
                color: #cccccc;
                border: 1px solid #3a3a3a;
                margin-top: 1ex;
                border-radius: 4px;
                background-color: #252526;
            }
        """)
        metrics_layout = QVBoxLayout()

        # Заголовки таблицы
        metrics_header_layout = QHBoxLayout()
        metrics_header_layout.addWidget(QLabel("Метрика"))
        metrics_header_layout.addWidget(QLabel("Модель 1"))
        metrics_header_layout.addWidget(QLabel("Модель 2"))
        metrics_header_layout.addStretch()

        for label in metrics_header_layout.children():
            if isinstance(label, QLabel):
                label.setStyleSheet("color: #cccccc; font-weight: bold; padding: 5px;")
                label.setFixedWidth(150)

        metrics_layout.addLayout(metrics_header_layout)

        # Создаем строки для метрик
        self.metrics_rows = {}
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-score"]

        for metric in metrics_names:
            row_layout = QHBoxLayout()

            metric_label = QLabel(metric)
            metric_label.setStyleSheet("color: #cccccc; padding: 5px;")
            metric_label.setFixedWidth(150)
            row_layout.addWidget(metric_label)

            model1_value = QLabel("—")
            model1_value.setStyleSheet("color: #4a90e2; padding: 5px; font-family: monospace;")
            model1_value.setFixedWidth(150)
            row_layout.addWidget(model1_value)

            model2_value = QLabel("—")
            model2_value.setStyleSheet("color: #e67e22; padding: 5px; font-family: monospace;")
            model2_value.setFixedWidth(150)
            row_layout.addWidget(model2_value)

            row_layout.addStretch()
            metrics_layout.addLayout(row_layout)

            self.metrics_rows[metric] = (model1_value, model2_value)

        # Разделитель
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #3a3a3a; max-height: 1px;")
        metrics_layout.addWidget(separator)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        return widget

    def load_plots(self):
        """Загрузка графиков и метрик для сравнения"""
        self.plot_figure.clear()

        # Устанавливаем темный фон для фигуры
        self.plot_figure.patch.set_facecolor('#1e1e1e')

        # Получаем выбранные данные
        model1_data = self.plot_model1_combo.currentData()
        model2_data = self.plot_model2_combo.currentData()

        # Очищаем таблицу метрик
        for metric in self.metrics_rows:
            model1_label, model2_label = self.metrics_rows[metric]
            model1_label.setText("—")
            model2_label.setText("—")

        # Обновляем метрики для модели 1
        if model1_data:
            name1 = model1_data.get('model_name', 'Модель 1')
            test_metrics = model1_data.get('test_metrics', {})

            if test_metrics:
                for metric, (label, _) in self.metrics_rows.items():
                    value = test_metrics.get(metric.lower(), None)
                    if value is not None:
                        label.setText(f"{value:.4f}")
            else:
                history = model1_data.get('history', {})
                best_val_acc = history.get('best_val_acc', 0)
                self.metrics_rows["Accuracy"][0].setText(f"{best_val_acc:.2f}%")
                self.metrics_rows["Precision"][0].setText("—")
                self.metrics_rows["Recall"][0].setText("—")
                self.metrics_rows["F1-score"][0].setText("—")

        # Обновляем метрики для модели 2
        if model2_data:
            name2 = model2_data.get('model_name', 'Модель 2')
            test_metrics = model2_data.get('test_metrics', {})

            if test_metrics:
                for metric, (_, label) in self.metrics_rows.items():
                    value = test_metrics.get(metric.lower(), None)
                    if value is not None:
                        label.setText(f"{value:.4f}")
            else:
                history = model2_data.get('history', {})
                best_val_acc = history.get('best_val_acc', 0)
                self.metrics_rows["Accuracy"][1].setText(f"{best_val_acc:.2f}%")
                self.metrics_rows["Precision"][1].setText("—")
                self.metrics_rows["Recall"][1].setText("—")
                self.metrics_rows["F1-score"][1].setText("—")

        # Отрисовка графиков
        ax1 = self.plot_figure.add_subplot(121)
        ax2 = self.plot_figure.add_subplot(122)

        # Устанавливаем темный фон для осей
        ax1.set_facecolor('#2b2b2b')
        ax2.set_facecolor('#2b2b2b')

        has_any_data = False

        if model1_data:
            history = model1_data.get('history')
            name = model1_data.get('model_name', 'Модель 1')

            if history and 'train_loss' in history and len(history['train_loss']) > 0:
                has_any_data = True
                epochs = range(1, len(history['train_loss']) + 1)
                ax1.plot(epochs, history['train_loss'], label=f'{name} Train', color='#4a90e2', linestyle='-',
                         linewidth=2)
                ax1.plot(epochs, history['val_loss'], label=f'{name} Val', color='#4a90e2', linestyle='--', linewidth=2)
                ax2.plot(epochs, history['train_acc'], label=f'{name} Train', color='#4a90e2', linestyle='-',
                         linewidth=2)
                ax2.plot(epochs, history['val_acc'], label=f'{name} Val', color='#4a90e2', linestyle='--', linewidth=2)

        if model2_data:
            history = model2_data.get('history')
            name = model2_data.get('model_name', 'Модель 2')

            if history and 'train_loss' in history and len(history['train_loss']) > 0:
                has_any_data = True
                epochs = range(1, len(history['train_loss']) + 1)
                ax1.plot(epochs, history['train_loss'], label=f'{name} Train', color='#e67e22', linestyle='-',
                         linewidth=2)
                ax1.plot(epochs, history['val_loss'], label=f'{name} Val', color='#e67e22', linestyle='--', linewidth=2)
                ax2.plot(epochs, history['train_acc'], label=f'{name} Train', color='#e67e22', linestyle='-',
                         linewidth=2)
                ax2.plot(epochs, history['val_acc'], label=f'{name} Val', color='#e67e22', linestyle='--', linewidth=2)

        if not has_any_data and not model1_data and not model2_data:
            ax1.text(0.5, 0.5, 'Выберите модели для сравнения', ha='center', va='center', fontsize=12, color='gray')
            ax2.text(0.5, 0.5, 'Выберите модели для сравнения', ha='center', va='center', fontsize=12, color='gray')

        ax1.set_xlabel('Epoch', color='#cccccc')
        ax1.set_ylabel('Loss', color='#cccccc')
        ax1.set_title('График потерь', color='#cccccc')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(colors='#cccccc')

        ax2.set_xlabel('Epoch', color='#cccccc')
        ax2.set_ylabel('Accuracy (%)', color='#cccccc')
        ax2.set_title('График точности', color='#cccccc')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(colors='#cccccc')

        self.plot_figure.tight_layout()
        self.plot_canvas.draw()

    def refresh_plot_combos(self):
        """Обновление списков моделей в графиках с проверкой существования"""
        self.plot_model1_combo.clear()
        self.plot_model2_combo.clear()
        self.plot_model1_combo.addItem("-- Выберите модель --", None)
        self.plot_model2_combo.addItem("-- Выберите модель --", None)

        # Получаем доступные модели из конфига
        available_models = config.get_available_models()

        print(f"DEBUG: Найдено моделей: {len(available_models)}")

        for model in available_models:
            if model['log'] is not None:
                log = model['log']
                if 'history' in log and log['history']:
                    history = log['history']
                    if 'train_loss' in history and len(history['train_loss']) > 0:
                        name = f"{model['model_name']} ({log['model_type']}) - {log['timestamp'][:16]}"
                        # ВАЖНО: передаем сам log, а не None
                        self.plot_model1_combo.addItem(name, log)
                        self.plot_model2_combo.addItem(name, log)
                        print(f"DEBUG: Добавлена модель: {name}")

    def load_available_models(self):
        """Загрузка списка доступных моделей"""
        self.mfcc_models.clear()
        self.yamnet_models.clear()

        # Получаем доступные модели
        available_models = config.get_available_models()

        for model in available_models:
            name = model['model_name']
            path = model['model_path']

            if 'mfcc' in name.lower():
                self.mfcc_models.append((name, path))
            elif 'yamnet' in name.lower():
                self.yamnet_models.append((name, path))

        if hasattr(self, 'mfcc_model_combo'):
            self.update_model_combos()

        self.refresh_plot_combos()

    def update_model_combos(self):
        """Обновление списков моделей в комбобоксах"""
        self.mfcc_model_combo.clear()
        self.mfcc_model_combo.addItem("-- Не использовать --", None)
        for name, path in self.mfcc_models:
            self.mfcc_model_combo.addItem(name, path)

        self.micro_mfcc_combo.clear()
        self.micro_mfcc_combo.addItem("-- Не использовать --", None)
        for name, path in self.mfcc_models:
            self.micro_mfcc_combo.addItem(name, path)

        self.yamnet_model_combo.clear()
        self.yamnet_model_combo.addItem("-- Не использовать --", None)
        for name, path in self.yamnet_models:
            self.yamnet_model_combo.addItem(name, path)

        self.micro_yamnet_combo.clear()
        self.micro_yamnet_combo.addItem("-- Не использовать --", None)
        for name, path in self.yamnet_models:
            self.micro_yamnet_combo.addItem(name, path)

        # Обновляем названия текущих моделей
        self.current_mfcc_model_name = self.mfcc_model_combo.currentText()
        self.current_yamnet_model_name = self.yamnet_model_combo.currentText()

    def create_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Основная горизонтальная компоновка
        main_h_layout = QHBoxLayout()

        # Левая колонка - параметры
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)

        # Параметры обучения
        params_group = QGroupBox("Параметры обучения")
        params_layout = QGridLayout()

        # Пути к данным
        params_layout.addWidget(QLabel("Папка с дронами:"), 0, 0)
        self.drone_path_edit = QLineEdit()
        self.drone_path_edit.setPlaceholderText("Выберите папку со звуками дронов")
        params_layout.addWidget(self.drone_path_edit, 0, 1)
        drone_browse_btn = QPushButton("Обзор")
        drone_browse_btn.clicked.connect(lambda: self.browse_folder(self.drone_path_edit))
        params_layout.addWidget(drone_browse_btn, 0, 2)

        params_layout.addWidget(QLabel("Папка с шумами:"), 1, 0)
        self.noise_path_edit = QLineEdit()
        self.noise_path_edit.setPlaceholderText("Выберите папку со звуками шумов")
        params_layout.addWidget(self.noise_path_edit, 1, 1)
        noise_browse_btn = QPushButton("Обзор")
        noise_browse_btn.clicked.connect(lambda: self.browse_folder(self.noise_path_edit))
        params_layout.addWidget(noise_browse_btn, 1, 2)

        # Разделитель
        params_layout.addWidget(QLabel(""), 2, 0, 1, 3)

        # MFCC модель
        self.train_mfcc_check = QCheckBox("Обучать MFCC+ модель")
        self.train_mfcc_check.setChecked(True)
        self.train_mfcc_check.toggled.connect(self.on_mfcc_toggled)
        params_layout.addWidget(self.train_mfcc_check, 3, 0, 1, 2)

        params_layout.addWidget(QLabel("Название модели MFCC+:"), 4, 0)
        self.mfcc_name_edit = QLineEdit()
        self.mfcc_name_edit.setPlaceholderText("mfcc_model")
        self.mfcc_name_edit.setText("mfcc_model")
        params_layout.addWidget(self.mfcc_name_edit, 4, 1)

        # YAMNet модель
        self.train_yamnet_check = QCheckBox("Обучать YAMNet модель")
        self.train_yamnet_check.setChecked(True)
        self.train_yamnet_check.toggled.connect(self.on_yamnet_toggled)
        params_layout.addWidget(self.train_yamnet_check, 5, 0, 1, 2)

        params_layout.addWidget(QLabel("Название модели YAMNet:"), 6, 0)
        self.yamnet_name_edit = QLineEdit()
        self.yamnet_name_edit.setPlaceholderText("yamnet_model")
        self.yamnet_name_edit.setText("yamnet_model")
        params_layout.addWidget(self.yamnet_name_edit, 6, 1)

        # Разделитель
        params_layout.addWidget(QLabel(""), 7, 0, 1, 3)

        # Параметры - переносим вправо, добавляя колонки
        params_layout.addWidget(QLabel("Эпохи:"), 8, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 10000)
        self.epochs_spin.setValue(config.DEFAULT_EPOCHS)
        params_layout.addWidget(self.epochs_spin, 8, 1)

        params_layout.addWidget(QLabel("Batch size:"), 9, 0)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(2, 128)
        self.batch_spin.setValue(config.DEFAULT_BATCH_SIZE)
        self.batch_spin.setToolTip("Минимальное значение 2 (требование BatchNorm слоев)")
        params_layout.addWidget(self.batch_spin, 9, 1)

        # Параметры - learning rate
        params_layout.addWidget(QLabel("Learning rate:"), 10, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setSingleStep(0.0001)  # Шаг 0.0001
        self.lr_spin.setValue(0.001)  # Явно устанавливаем 0.001
        self.lr_spin.setDecimals(6)
        self.lr_spin.setToolTip("Диапазон: 0.00001 - 1.0")
        params_layout.addWidget(self.lr_spin, 10, 1)

        # Разбиение данных
        params_layout.addWidget(QLabel("Train %:"), 11, 0)
        self.train_spin = QSpinBox()
        self.train_spin.setRange(50, 90)
        self.train_spin.setValue(config.DEFAULT_TRAIN_RATIO)
        self.train_spin.valueChanged.connect(self.update_split)
        params_layout.addWidget(self.train_spin, 11, 1)

        params_layout.addWidget(QLabel("Val %:"), 12, 0)
        self.val_spin = QSpinBox()
        self.val_spin.setRange(5, 40)
        self.val_spin.setValue(config.DEFAULT_VAL_RATIO)
        self.val_spin.valueChanged.connect(self.update_split)
        params_layout.addWidget(self.val_spin, 12, 1)

        params_layout.addWidget(QLabel("Test %:"), 13, 0)
        self.test_spin = QSpinBox()
        self.test_spin.setRange(5, 40)
        self.test_spin.setValue(config.DEFAULT_TEST_RATIO)
        self.test_spin.valueChanged.connect(self.update_split)
        params_layout.addWidget(self.test_spin, 13, 1)

        self.split_info_label = QLabel()
        self.update_split()
        params_layout.addWidget(self.split_info_label, 14, 1)

        params_group.setLayout(params_layout)
        left_layout.addWidget(params_group)

        # Кнопки
        btn_layout = QHBoxLayout()
        self.start_train_btn = QPushButton("🚁 Начать обучение")
        self.start_train_btn.clicked.connect(self.start_training)
        self.start_train_btn.setStyleSheet("background-color: #4CAF50; font-size: 14px; padding: 10px;")
        btn_layout.addWidget(self.start_train_btn)

        self.stop_train_btn = QPushButton("⏹ Остановить")
        self.stop_train_btn.clicked.connect(self.stop_training)
        self.stop_train_btn.setEnabled(False)
        self.stop_train_btn.setStyleSheet("background-color: #f44336;")
        btn_layout.addWidget(self.stop_train_btn)
        left_layout.addLayout(btn_layout)

        left_layout.addStretch()

        # Правая колонка - лог
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)

        log_group = QGroupBox("Лог обучения")
        log_layout = QVBoxLayout()
        self.training_log = QTextEdit()
        self.training_log.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: 'Courier New';
                border: 1px solid #3a3a3a;
            }
        """)
        self.training_log.setReadOnly(True)
        log_layout.addWidget(self.training_log)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)

        # Добавляем колонки в основную компоновку
        main_h_layout.addWidget(left_column, 40)
        main_h_layout.addWidget(right_column, 60)

        layout.addLayout(main_h_layout)

        # Графики внизу
        plot_group = QGroupBox("Графики текущего обучения")
        plot_layout = QHBoxLayout()

        # Создаем фигуру с явным указанием фона
        self.figure = plt.figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        # Устанавливаем начальное сообщение
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)

        # Устанавливаем фон для осей
        ax1.set_facecolor('#2b2b2b')
        ax2.set_facecolor('#2b2b2b')

        # Добавляем текст
        ax1.text(0.5, 0.5, 'Обучение не запущено', ha='center', va='center', fontsize=12, color='#cccccc')
        ax2.text(0.5, 0.5, 'Обучение не запущено', ha='center', va='center', fontsize=12, color='#cccccc')

        # Устанавливаем границы и убираем оси
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

        self.canvas.draw()

        return widget

    def on_mfcc_toggled(self, checked):
        self.mfcc_name_edit.setEnabled(checked)

    def on_yamnet_toggled(self, checked):
        self.yamnet_name_edit.setEnabled(checked)

    def create_file_detection_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Выбор файлов
        file_group = QGroupBox("Выбор аудиофайлов")
        file_layout = QVBoxLayout()

        path_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Выберите файл или папку с аудио")
        path_layout.addWidget(self.file_path_edit)

        file_browse_btn = QPushButton("Обзор файла")
        file_browse_btn.clicked.connect(lambda: self.browse_file(self.file_path_edit))
        path_layout.addWidget(file_browse_btn)

        folder_browse_btn = QPushButton("Обзор папки")
        folder_browse_btn.clicked.connect(lambda: self.browse_folder(self.file_path_edit))
        path_layout.addWidget(folder_browse_btn)

        file_layout.addLayout(path_layout)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Выбор моделей
        models_group = QGroupBox("Выбор моделей")
        models_layout = QGridLayout()

        models_layout.addWidget(QLabel("MFCC+ модель:"), 0, 0)
        self.mfcc_model_combo = QComboBox()
        self.mfcc_model_combo.addItem("-- Не использовать --", None)
        models_layout.addWidget(self.mfcc_model_combo, 0, 1)

        # Добавляем подсказку о выбранной модели
        self.mfcc_model_combo.currentTextChanged.connect(self.update_mfcc_model_label)

        models_layout.addWidget(QLabel("YAMNet модель:"), 1, 0)
        self.yamnet_model_combo = QComboBox()
        self.yamnet_model_combo.addItem("-- Не использовать --", None)
        models_layout.addWidget(self.yamnet_model_combo, 1, 1)

        self.yamnet_model_combo.currentTextChanged.connect(self.update_yamnet_model_label)

        models_group.setLayout(models_layout)
        layout.addWidget(models_group)

        # Кнопки
        btn_layout = QHBoxLayout()
        self.start_detection_btn = QPushButton("🎯 Запустить детекцию")
        self.start_detection_btn.clicked.connect(self.start_file_detection)
        btn_layout.addWidget(self.start_detection_btn)

        self.save_results_btn = QPushButton("💾 Сохранить результаты")
        self.save_results_btn.clicked.connect(self.save_detection_results)
        self.save_results_btn.setEnabled(False)
        btn_layout.addWidget(self.save_results_btn)

        self.clear_results_btn = QPushButton("🗑️ Очистить результаты")
        self.clear_results_btn.clicked.connect(self.clear_detection_results)
        self.clear_results_btn.setStyleSheet("background-color: #f44336;")
        btn_layout.addWidget(self.clear_results_btn)

        layout.addLayout(btn_layout)

        # Таблица результатов (добавляем колонки с названиями моделей)
        self.results_table = QTableWidget()
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: #252526;
                color: #cccccc;
                gridline-color: #3a3a3a;
                border: 1px solid #3a3a3a;
                alternate-background-color: #2d2d2d;
            }
            QTableWidget::item {
                padding: 5px;
            }
            QTableWidget::item:selected {
                background-color: #3c3c3c;
            }
            QHeaderView::section {
                background-color: #2d2d2d;
                color: #cccccc;
                padding: 5px;
                border: 1px solid #3a3a3a;
            }
            QTableCornerButton::section {
                background-color: #2d2d2d;
                border: 1px solid #3a3a3a;
            }
        """)
        self.results_table.setColumnCount(7)  # Увеличиваем до 7 колонок
        self.results_table.setHorizontalHeaderLabels([
            "Файл",
            "MFCC+ модель", "MFCC+ результат",
            "YAMNet модель", "YAMNet результат",
            "Длина (с)", "Статус"
        ])
        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        layout.addWidget(self.results_table)

        # Сохраняем названия выбранных моделей
        self.current_mfcc_model_name = "-- Не использовать --"
        self.current_yamnet_model_name = "-- Не использовать --"

        return widget

    def update_mfcc_model_label(self, text):
        """Обновление названия выбранной MFCC модели"""
        self.current_mfcc_model_name = text

    def update_yamnet_model_label(self, text):
        """Обновление названия выбранной YAMNet модели"""
        self.current_yamnet_model_name = text

    def clear_detection_results(self):
        """Очистка таблицы результатов"""
        reply = QMessageBox.question(
            self,
            "Подтверждение",
            "Очистить все результаты детекции?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.results_table.setRowCount(0)
            self.last_results = None
            self.save_results_btn.setEnabled(False)
            self.statusBar().showMessage("Результаты очищены")

    def create_microphone_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Выбор моделей
        models_group = QGroupBox("Выбор моделей для детекции")
        models_layout = QGridLayout()

        models_layout.addWidget(QLabel("MFCC+ модель:"), 0, 0)
        self.micro_mfcc_combo = QComboBox()
        self.micro_mfcc_combo.addItem("-- Не использовать --")
        models_layout.addWidget(self.micro_mfcc_combo, 0, 1)

        models_layout.addWidget(QLabel("YAMNet модель:"), 1, 0)
        self.micro_yamnet_combo = QComboBox()
        self.micro_yamnet_combo.addItem("-- Не использовать --")
        models_layout.addWidget(self.micro_yamnet_combo, 1, 1)

        models_group.setLayout(models_layout)
        layout.addWidget(models_group)

        # Кнопки
        btn_layout = QHBoxLayout()
        self.start_micro_btn = QPushButton("🎤 Начать детекцию с микрофона")
        self.start_micro_btn.clicked.connect(self.start_micro_detection)
        self.start_micro_btn.setStyleSheet("background-color: #4CAF50;")
        btn_layout.addWidget(self.start_micro_btn)

        self.stop_micro_btn = QPushButton("⏹ Остановить")
        self.stop_micro_btn.clicked.connect(self.stop_micro_detection)
        self.stop_micro_btn.setEnabled(False)
        self.stop_micro_btn.setStyleSheet("background-color: #f44336;")
        btn_layout.addWidget(self.stop_micro_btn)
        layout.addLayout(btn_layout)

        # Индикатор уровня звука
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        layout.addWidget(self.level_bar)

        # Результаты
        self.micro_results = QTextEdit()
        self.micro_results.setReadOnly(True)
        self.micro_results.setMaximumHeight(150)
        layout.addWidget(self.micro_results)

        # График спектра - создаем фигуру с темным фоном
        self.micro_figure = plt.figure(figsize=(10, 4), facecolor='#1e1e1e')
        self.micro_canvas = FigureCanvas(self.micro_figure)
        layout.addWidget(self.micro_canvas)

        # Устанавливаем начальное сообщение
        self.micro_figure.clear()
        ax = self.micro_figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.text(0.5, 0.5, 'Нажмите "Начать детекцию"',
                ha='center', va='center', fontsize=12, color='#cccccc')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.micro_canvas.draw()

        return widget

    def update_split(self):
        train = self.train_spin.value()
        val = self.val_spin.value()
        test = self.test_spin.value()
        total = train + val + test

        if total != 100:
            self.split_info_label.setText(f"⚠️ Сумма должна быть 100% (сейчас {total}%)")
            self.split_info_label.setStyleSheet("color: orange;")
        else:
            self.split_info_label.setText(f"Train: {train}%, Val: {val}%, Test: {test}%")
            self.split_info_label.setStyleSheet("color: #aaa;")

    def browse_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку")
        if folder:
            line_edit.setText(folder)

    def browse_file(self, line_edit):
        file, _ = QFileDialog.getOpenFileName(
            self, "Выберите аудиофайл", "",
            "Аудиофайлы (*.wav *.mp3 *.ogg *.flac)"
        )
        if file:
            line_edit.setText(file)

    def start_training(self):
        if not self.drone_path_edit.text() or not self.noise_path_edit.text():
            QMessageBox.warning(self, "Предупреждение", "Выберите папки с данными!")
            return

        train_mfcc = self.train_mfcc_check.isChecked()
        train_yamnet = self.train_yamnet_check.isChecked()

        if not train_mfcc and not train_yamnet:
            QMessageBox.warning(self, "Предупреждение", "Выберите хотя бы одну модель для обучения!")
            return

        mfcc_name = self.mfcc_name_edit.text().strip() if train_mfcc else ""
        yamnet_name = self.yamnet_name_edit.text().strip() if train_yamnet else ""

        if train_mfcc and not mfcc_name:
            mfcc_name = f"mfcc_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}"
        if train_yamnet and not yamnet_name:
            yamnet_name = f"yamnet_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}"

        train_total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        if train_total != 100:
            QMessageBox.warning(self, "Предупреждение", "Сумма Train + Val + Test должна быть 100%!")
            return

        params = {
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'learning_rate': self.lr_spin.value(),
            'train_ratio': self.train_spin.value() / 100,
            'val_ratio': self.val_spin.value() / 100,
            'test_ratio': self.test_spin.value() / 100
        }

        self.start_train_btn.setEnabled(False)
        self.stop_train_btn.setEnabled(True)
        self.training_log.clear()

        self.training_thread = TrainingThread(
            self.drone_path_edit.text(),
            self.noise_path_edit.text(),
            params,
            train_mfcc,
            train_yamnet,
            mfcc_name,
            yamnet_name
        )
        self.training_thread.progress_update.connect(self.update_training_log)
        self.training_thread.training_finished.connect(self.training_finished)
        self.training_thread.start()

        # ОЧИЩАЕМ ГРАФИК ПЕРЕД НОВЫМ ОБУЧЕНИЕМ
        self.figure.clear()
        ax1 = self.figure.add_subplot(121)
        ax2 = self.figure.add_subplot(122)
        ax1.text(0.5, 0.5, 'Обучение запущено...', ha='center', va='center', fontsize=12)
        ax2.text(0.5, 0.5, 'Обучение запущено...', ha='center', va='center', fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        self.canvas.draw()

    def stop_training(self):
        """Остановка обучения"""
        self.stop_train_btn.setEnabled(False)
        self.update_training_log("⚠️ Остановка обучения недоступна. Дождитесь завершения.")

    def update_training_log(self, message, new_line=True):
        """Обновление лога с возможностью перезаписи последней строки"""
        if new_line:
            # Обычное добавление новой строки
            self.training_log.append(message)
        else:
            # Перезаписываем последнюю строку
            cursor = self.training_log.textCursor()
            cursor.movePosition(cursor.End)
            # Выделяем последнюю строку
            cursor.movePosition(cursor.StartOfLine, cursor.KeepAnchor)
            cursor.removeSelectedText()
            # Вставляем новое сообщение
            cursor.insertText(message)
            # Перемещаем курсор в конец
            cursor.movePosition(cursor.End)
            self.training_log.setTextCursor(cursor)

        # Прокручиваем вниз
        self.training_log.verticalScrollBar().setValue(
            self.training_log.verticalScrollBar().maximum()
        )

    def training_finished(self, mfcc_history, yamnet_history):
        print("=" * 50)
        print("TRAINING_FINISHED: Начало")
        print(f"mfcc_history: {mfcc_history is not None}")
        print(f"yamnet_history: {yamnet_history is not None}")

        self.mfcc_history = mfcc_history
        self.yamnet_history = yamnet_history

        self.start_train_btn.setEnabled(True)
        self.stop_train_btn.setEnabled(False)

        print("Обновление списка моделей...")
        self.load_available_models()
        print("Список моделей обновлен")

        self.update_training_log("\n✅ Обучение завершено!")
        print("Лог обновлен")

        # Сохраняем графики (только для обученных моделей)
        if mfcc_history is not None:
            print("Начинаем обработку MFCC истории...")
            try:
                from utils.plot_utils import plot_training_history
                print("plot_training_history импортирован")

                # Проверяем, что папка для графиков существует
                os.makedirs(config.PLOTS_DIR, exist_ok=True)
                print(f"Папка для графиков: {config.PLOTS_DIR}")

                # Получаем имя модели - только если MFCC обучалась
                if hasattr(self.training_thread, 'mfcc_name') and self.training_thread.mfcc_name:
                    model_name = self.training_thread.mfcc_name
                else:
                    model_name = "mfcc_model"
                print(f"MFCC model_name: {model_name}")

                plot_path = os.path.join(config.PLOTS_DIR, f"{model_name}_training_history.png")
                print(f"Путь для сохранения: {plot_path}")

                # Проверяем, что в истории есть данные для графика
                print(f"train_loss in history: {'train_loss' in mfcc_history}")
                if 'train_loss' in mfcc_history:
                    print(f"len train_loss: {len(mfcc_history['train_loss'])}")

                if 'train_loss' in mfcc_history and mfcc_history['train_loss'] and len(mfcc_history['train_loss']) > 0:
                    print("Начинаем сохранение графика MFCC...")
                    try:
                        result = plot_training_history(mfcc_history, title=model_name, save_path=plot_path)
                        print(f"Результат сохранения: {result}")
                        self.update_training_log(f"📊 График MFCC+ сохранен: {plot_path}")
                    except Exception as plot_error:
                        print(f"ОШИБКА ПРИ СОЗДАНИИ ГРАФИКА: {plot_error}")
                        import traceback
                        traceback.print_exc()
                        self.update_training_log(f"⚠️ Ошибка при создании графика MFCC+: {plot_error}")
                else:
                    print("Нет данных для графика MFCC")
                    self.update_training_log(f"⚠️ Нет данных для графика MFCC+")
            except Exception as e:
                print(f"ОШИБКА В БЛОКЕ MFCC: {e}")
                import traceback
                traceback.print_exc()
                self.update_training_log(f"⚠️ Ошибка сохранения графика MFCC+: {e}")

            # Проверяем наличие best_val_acc перед выводом
            if 'best_val_acc' in mfcc_history:
                self.update_training_log(f"MFCC+ лучшая точность: {mfcc_history['best_val_acc']:.2f}%")
            print("MFCC обработка завершена")

        if yamnet_history is not None:
            print("Начинаем обработку YAMNet истории...")
            try:
                from utils.plot_utils import plot_training_history
                print("plot_training_history импортирован")

                # Проверяем, что папка для графиков существует
                os.makedirs(config.PLOTS_DIR, exist_ok=True)
                print(f"Папка для графиков: {config.PLOTS_DIR}")

                # Получаем имя модели - только если YAMNet обучалась
                if hasattr(self.training_thread, 'yamnet_name') and self.training_thread.yamnet_name:
                    model_name = self.training_thread.yamnet_name
                else:
                    model_name = "yamnet_model"
                print(f"YAMNet model_name: {model_name}")

                plot_path = os.path.join(config.PLOTS_DIR, f"{model_name}_training_history.png")
                print(f"Путь для сохранения: {plot_path}")

                # Проверяем, что в истории есть данные для графика
                print(f"train_loss in history: {'train_loss' in yamnet_history}")
                if 'train_loss' in yamnet_history:
                    print(f"len train_loss: {len(yamnet_history['train_loss'])}")

                if 'train_loss' in yamnet_history and yamnet_history['train_loss'] and len(
                        yamnet_history['train_loss']) > 0:
                    print("Начинаем сохранение графика YAMNet...")
                    try:
                        result = plot_training_history(yamnet_history, title=model_name, save_path=plot_path)
                        print(f"Результат сохранения: {result}")
                        self.update_training_log(f"📊 График YAMNet сохранен: {plot_path}")
                    except Exception as plot_error:
                        print(f"ОШИБКА ПРИ СОЗДАНИИ ГРАФИКА: {plot_error}")
                        import traceback
                        traceback.print_exc()
                        self.update_training_log(f"⚠️ Ошибка при создании графика YAMNet: {plot_error}")
                else:
                    print("Нет данных для графика YAMNet")
                    self.update_training_log(f"⚠️ Нет данных для графика YAMNet")
            except Exception as e:
                print(f"ОШИБКА В БЛОКЕ YAMNet: {e}")
                import traceback
                traceback.print_exc()
                self.update_training_log(f"⚠️ Ошибка сохранения графика YAMNet: {e}")

            # Проверяем наличие best_val_acc перед выводом
            if 'best_val_acc' in yamnet_history:
                self.update_training_log(f"YAMNet лучшая точность: {yamnet_history['best_val_acc']:.2f}%")
            print("YAMNet обработка завершена")

        print("Обновляем графики в GUI...")
        # Обновляем графики в GUI - передаем только те, которые были обучены
        try:
            self.update_training_plots(mfcc_history if mfcc_history is not None else None,
                                       yamnet_history if yamnet_history is not None else None)
            print("Графики обновлены")
        except Exception as e:
            print(f"ОШИБКА ПРИ ОБНОВЛЕНИИ ГРАФИКОВ: {e}")
            import traceback
            traceback.print_exc()

        print("TRAINING_FINISHED: Завершение")
        print("=" * 50)

        QApplication.processEvents()

    def update_training_plots(self, mfcc_history, yamnet_history):
        """Обновление графиков в GUI - только если есть данные"""
        if mfcc_history is None and yamnet_history is None:
            return

        try:
            self.figure.clear()

            ax1 = self.figure.add_subplot(121)
            ax2 = self.figure.add_subplot(122)

            ax1.set_facecolor('#2b2b2b')
            ax2.set_facecolor('#2b2b2b')

            has_any_data = False

            # Отображаем MFCC данные, если они есть
            if mfcc_history is not None and mfcc_history.get('train_loss') and len(mfcc_history['train_loss']) > 0:
                has_any_data = True
                epochs = range(1, len(mfcc_history['train_loss']) + 1)
                ax1.plot(epochs, mfcc_history['train_loss'], label='MFCC+ Train', color='#4a90e2', linestyle='-')
                ax1.plot(epochs, mfcc_history['val_loss'], label='MFCC+ Val', color='#4a90e2', linestyle='--')
                ax2.plot(epochs, mfcc_history['train_acc'], label='MFCC+ Train', color='#4a90e2', linestyle='-')
                ax2.plot(epochs, mfcc_history['val_acc'], label='MFCC+ Val', color='#4a90e2', linestyle='--')

            # Отображаем YAMNet данные, если они есть
            if yamnet_history is not None and yamnet_history.get('train_loss') and len(
                    yamnet_history['train_loss']) > 0:
                has_any_data = True
                epochs = range(1, len(yamnet_history['train_loss']) + 1)
                ax1.plot(epochs, yamnet_history['train_loss'], label='YAMNet Train', color='#e67e22', linestyle='-')
                ax1.plot(epochs, yamnet_history['val_loss'], label='YAMNet Val', color='#e67e22', linestyle='--')
                ax2.plot(epochs, yamnet_history['train_acc'], label='YAMNet Train', color='#e67e22', linestyle='-')
                ax2.plot(epochs, yamnet_history['val_acc'], label='YAMNet Val', color='#e67e22', linestyle='--')

            # Если нет данных для отображения
            if not has_any_data:
                ax1.text(0.5, 0.5, 'Нет данных для отображения', ha='center', va='center', fontsize=12, color='#cccccc')
                ax2.text(0.5, 0.5, 'Нет данных для отображения', ha='center', va='center', fontsize=12, color='#cccccc')
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax2.set_xticks([])
                ax2.set_yticks([])
            else:
                ax1.set_xlabel('Epoch', color='#cccccc')
                ax1.set_ylabel('Loss', color='#cccccc')
                ax1.set_title('График потерь', color='#cccccc')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.tick_params(colors='#cccccc')

                ax2.set_xlabel('Epoch', color='#cccccc')
                ax2.set_ylabel('Accuracy (%)', color='#cccccc')
                ax2.set_title('График точности', color='#cccccc')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(colors='#cccccc')

            self.figure.tight_layout()
            self.canvas.draw()
            self.canvas.flush_events()
        except Exception as e:
            print(f"Ошибка обновления графиков: {e}")
            import traceback
            traceback.print_exc()

    def start_file_detection(self):
        path = self.file_path_edit.text()
        if not path:
            QMessageBox.warning(self, "Предупреждение", "Выберите файл или папку!")
            return

        files = []
        if os.path.isfile(path):
            files = [path]
        elif os.path.isdir(path):
            for ext in ['*.wav', '*.mp3', '*.ogg', '*.flac']:
                files.extend(glob.glob(os.path.join(path, ext)))

        if not files:
            QMessageBox.warning(self, "Предупреждение", "Не найдено аудиофайлов!")
            return

        mfcc_path = self.mfcc_model_combo.currentData()
        yamnet_path = self.yamnet_model_combo.currentData()

        # НЕ ОЧИЩАЕМ ТАБЛИЦУ, а добавляем разделитель позже
        self.detection_thread = DetectionThread(files, mfcc_path, yamnet_path)
        self.detection_thread.progress_update.connect(self.update_training_log)
        self.detection_thread.result_ready.connect(self.display_results)
        self.detection_thread.start()
        self.start_detection_btn.setEnabled(False)

    def display_results(self, results):
        # Получаем названия выбранных моделей
        mfcc_model_name = self.current_mfcc_model_name
        yamnet_model_name = self.current_yamnet_model_name

        # Добавляем разделитель, если уже есть результаты
        if self.results_table.rowCount() > 0:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            separator = QTableWidgetItem("═" * 50)
            separator.setBackground(QColor(60, 60, 60))
            separator.setForeground(QColor(100, 100, 100))
            separator.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row, 0, separator)
            self.results_table.setSpan(row, 0, 1, 7)  # Растягиваем на 7 колонок

            # Добавляем информационную строку с датой
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            date_item = QTableWidgetItem(f"📅 {QDateTime.currentDateTime().toString('dd.MM.yyyy hh:mm:ss')}")
            date_item.setBackground(QColor(50, 50, 50))
            date_item.setForeground(QColor(150, 150, 150))
            self.results_table.setItem(row, 0, date_item)
            self.results_table.setSpan(row, 0, 1, 7)

        # Добавляем новые результаты
        for result in results:
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            # Файл
            file_item = QTableWidgetItem(os.path.basename(result['file']))
            file_item.setToolTip(result['file'])
            self.results_table.setItem(row, 0, file_item)

            # MFCC+ модель (название модели)
            mfcc_model_item = QTableWidgetItem(mfcc_model_name)
            mfcc_model_item.setForeground(QColor(150, 150, 150))
            self.results_table.setItem(row, 1, mfcc_model_item)

            # MFCC+ результат
            if result['mfcc_prob'] is not None:
                mfcc_text = f"{result['mfcc_prob']:.3f} ({result['mfcc_result']})"
                mfcc_item = QTableWidgetItem(mfcc_text)
                if result['mfcc_result'] == "ДА (дрон)":
                    mfcc_item.setForeground(QColor(100, 255, 100))
                else:
                    mfcc_item.setForeground(QColor(255, 100, 100))
                self.results_table.setItem(row, 2, mfcc_item)
            else:
                self.results_table.setItem(row, 2, QTableWidgetItem("N/A"))

            # YAMNet модель (название модели)
            yamnet_model_item = QTableWidgetItem(yamnet_model_name)
            yamnet_model_item.setForeground(QColor(150, 150, 150))
            self.results_table.setItem(row, 3, yamnet_model_item)

            # YAMNet результат
            if result['yamnet_prob'] is not None:
                yamnet_text = f"{result['yamnet_prob']:.3f} ({result['yamnet_result']})"
                yamnet_item = QTableWidgetItem(yamnet_text)
                if result['yamnet_result'] == "ДА (дрон)":
                    yamnet_item.setForeground(QColor(100, 255, 100))
                else:
                    yamnet_item.setForeground(QColor(255, 100, 100))
                self.results_table.setItem(row, 4, yamnet_item)
            else:
                self.results_table.setItem(row, 4, QTableWidgetItem("N/A"))

            # Длина
            self.results_table.setItem(row, 5, QTableWidgetItem(f"{result['duration']:.1f}"))

            # Статус
            if result['mfcc_result'] == result['yamnet_result'] and result['mfcc_result'] != 'N/A':
                status = "✅ Совпадают"
                status_item = QTableWidgetItem(status)
                status_item.setForeground(QColor(100, 255, 100))
                self.results_table.setItem(row, 6, status_item)
            elif result['mfcc_result'] != result['yamnet_result'] and result['mfcc_result'] != 'N/A' and result[
                'yamnet_result'] != 'N/A':
                status = "⚠️ Расходятся"
                status_item = QTableWidgetItem(status)
                status_item.setForeground(QColor(255, 200, 100))
                self.results_table.setItem(row, 6, status_item)
            else:
                self.results_table.setItem(row, 6, QTableWidgetItem("❓ Н/Д"))

        # Прокручиваем к последней строке
        self.results_table.scrollToBottom()

        self.start_detection_btn.setEnabled(True)
        self.save_results_btn.setEnabled(True)

        # Сохраняем последние результаты для экспорта
        if hasattr(self, 'last_results') and self.last_results:
            self.last_results.extend(results)
        else:
            self.last_results = results

    def save_detection_results(self):
        if not hasattr(self, 'last_results'):
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты",
            os.path.join(config.RESULTS_DIR, "detection_results.txt"),
            "Text files (*.txt)"
        )

        if filename:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("РЕЗУЛЬТАТЫ ДЕТЕКЦИИ ДРОНОВ\n")
                f.write(f"Дата: {QDateTime.currentDateTime().toString('yyyy-MM-dd hh:mm:ss')}\n")
                f.write("=" * 100 + "\n\n")

                # Заголовки
                f.write(
                    f"{'Файл':<40} {'MFCC+ модель':<20} {'MFCC+ результат':<20} {'YAMNet модель':<20} {'YAMNet результат':<20} {'Длина':<8} {'Статус':<15}\n")
                f.write("-" * 143 + "\n")

                for r in self.last_results:
                    filename = os.path.basename(r['file'])

                    mfcc_model = self.current_mfcc_model_name if len(
                        self.current_mfcc_model_name) < 18 else self.current_mfcc_model_name[:15] + "..."
                    yamnet_model = self.current_yamnet_model_name if len(
                        self.current_yamnet_model_name) < 18 else self.current_yamnet_model_name[:15] + "..."

                    mfcc_result = f"{r['mfcc_prob']:.3f} ({r['mfcc_result']})" if r['mfcc_prob'] is not None else "N/A"
                    yamnet_result = f"{r['yamnet_prob']:.3f} ({r['yamnet_result']})" if r[
                                                                                            'yamnet_prob'] is not None else "N/A"

                    if r['mfcc_result'] == r['yamnet_result'] and r['mfcc_result'] != 'N/A':
                        status = "Совпадают"
                    elif r['mfcc_result'] != r['yamnet_result'] and r['mfcc_result'] != 'N/A' and r[
                        'yamnet_result'] != 'N/A':
                        status = "РАСХОДЯТСЯ"
                    else:
                        status = "Н/Д"

                    f.write(
                        f"{filename:<40} {mfcc_model:<20} {mfcc_result:<20} {yamnet_model:<20} {yamnet_result:<20} {r['duration']:<8.1f} {status:<15}\n")

                QMessageBox.information(self, "Успех", f"Результаты сохранены в {filename}")

    def start_micro_detection(self):
        mfcc_path = self.micro_mfcc_combo.currentData()
        yamnet_path = self.micro_yamnet_combo.currentData()

        if not mfcc_path and not yamnet_path:
            QMessageBox.warning(self, "Предупреждение", "Выберите хотя бы одну модель!")
            return

        self.micro_thread = MicroDetectionThread(mfcc_path, yamnet_path)
        self.micro_thread.result_ready.connect(self.update_micro_results)
        self.micro_thread.level_update.connect(self.level_bar.setValue)
        self.micro_thread.spectrum_ready.connect(self.update_micro_spectrum)
        self.micro_thread.start()

        self.start_micro_btn.setEnabled(False)
        self.stop_micro_btn.setEnabled(True)
        self.micro_results.clear()

    def stop_micro_detection(self):
        if self.micro_thread:
            self.micro_thread.stop()
            self.micro_thread.wait()
            self.micro_thread = None

        # Очищаем график с темным фоном
        self.micro_figure.clear()
        self.micro_figure.patch.set_facecolor('#1e1e1e')

        ax = self.micro_figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.text(0.5, 0.5, 'Детекция остановлена', ha='center', va='center', fontsize=14, color='#cccccc')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.micro_canvas.draw()

        self.level_bar.setValue(0)
        self.start_micro_btn.setEnabled(True)
        self.stop_micro_btn.setEnabled(False)

    def update_micro_results(self, results):
        timestamp = QDateTime.currentDateTime().toString('hh:mm:ss')
        self.micro_results.append(f"[{timestamp}]")
        if results['mfcc_prob'] is not None:
            self.micro_results.append(f"  MFCC+: {results['mfcc_prob']:.3f} - {results['mfcc_result']}")
        if results['yamnet_prob'] is not None:
            self.micro_results.append(f"  YAMNet: {results['yamnet_prob']:.3f} - {results['yamnet_result']}")
        self.micro_results.append("-" * 30)

        self.micro_results.verticalScrollBar().setValue(
            self.micro_results.verticalScrollBar().maximum()
        )

    def update_micro_spectrum(self, freq, spectrum):
        self.micro_figure.clear()
        self.micro_figure.patch.set_facecolor('#1e1e1e')

        ax = self.micro_figure.add_subplot(111)
        ax.set_facecolor('#2b2b2b')

        ax.semilogx(freq, spectrum, color='#4a90e2', linewidth=1)
        ax.fill_between(freq, spectrum, alpha=0.3, color='#4a90e2')
        ax.axvspan(100, 5000, alpha=0.2, color='#e67e22', label='Диапазон дронов')
        ax.set_xlabel('Частота (Гц)', color='#cccccc')
        ax.set_ylabel('Амплитуда', color='#cccccc')
        ax.set_title('Спектр звука в реальном времени', color='#cccccc')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(20, 8000)
        ax.tick_params(colors='#cccccc')

        self.micro_canvas.draw()