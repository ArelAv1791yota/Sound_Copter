# utils/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_history(history, title="Training History", save_path=None):
    """Построение графиков обучения"""
    print(f"plot_training_history: начало, save_path={save_path}")
    print(f"history keys: {history.keys() if history else 'None'}")

    try:
        # Создаем фигуру с явным указанием стиля
        print("Создаем фигуру...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Устанавливаем цвета для осей и текста
        fig.patch.set_facecolor('#1e1e1e')
        print("Фигура создана")

        # График потерь
        if 'train_loss' in history and history['train_loss']:
            print(f"Строим график потерь, эпох: {len(history['train_loss'])}")
            epochs = range(1, len(history['train_loss']) + 1)
            ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            if 'val_loss' in history and history['val_loss']:
                ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            ax1.set_title('Loss', color='#cccccc')
            ax1.set_xlabel('Epoch', color='#cccccc')
            ax1.set_ylabel('Loss', color='#cccccc')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(colors='#cccccc')
            ax1.set_facecolor('#2b2b2b')

        # График точности
        if 'train_acc' in history and history['train_acc']:
            print(f"Строим график точности, эпох: {len(history['train_acc'])}")
            epochs = range(1, len(history['train_acc']) + 1)
            ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
            if 'val_acc' in history and history['val_acc']:
                ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
            ax2.set_title('Accuracy', color='#cccccc')
            ax2.set_xlabel('Epoch', color='#cccccc')
            ax2.set_ylabel('Accuracy (%)', color='#cccccc')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(colors='#cccccc')
            ax2.set_facecolor('#2b2b2b')

        # Если данных нет, показываем сообщение
        if not history.get('train_loss') and not history.get('train_acc'):
            print("Нет данных для отображения")
            ax1.text(0.5, 0.5, 'Нет данных для отображения', ha='center', va='center',
                     fontsize=12, color='#cccccc', transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'Нет данных для отображения', ha='center', va='center',
                     fontsize=12, color='#cccccc', transform=ax2.transAxes)

        fig.suptitle(title, color='#cccccc')
        fig.tight_layout()
        print("График построен")

        # Сохраняем график
        if save_path:
            try:
                print(f"Сохраняем график в {save_path}")
                # Создаем директорию если не существует
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
                print(f"График сохранен успешно")
                plt.close(fig)
                return save_path
            except Exception as e:
                print(f"Ошибка сохранения графика: {e}")
                import traceback
                traceback.print_exc()
                plt.close(fig)
                return None
        else:
            print("Показываем график (без сохранения)")
            plt.show()
            return fig

    except Exception as e:
        print(f"ОШИБКА В plot_training_history: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_comparison(mfcc_history, yamnet_history, save_path=None):
    """Сравнительные графики для двух моделей"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1e1e1e')

    # График потерь
    axes[0].plot(mfcc_history['val_loss'], label='MFCC+ (Val)', color='#4a90e2', linewidth=2, linestyle='-')
    axes[0].plot(yamnet_history['val_loss'], label='YAMNet (Val)', color='#e67e22', linewidth=2, linestyle='-')
    axes[0].plot(mfcc_history['train_loss'], label='MFCC+ (Train)', color='#4a90e2', linewidth=1, linestyle='--',
                 alpha=0.7)
    axes[0].plot(yamnet_history['train_loss'], label='YAMNet (Train)', color='#e67e22', linewidth=1, linestyle='--',
                 alpha=0.7)
    axes[0].set_xlabel('Epoch', color='#cccccc')
    axes[0].set_ylabel('Loss', color='#cccccc')
    axes[0].set_title('Сравнение потерь', color='#cccccc')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(colors='#cccccc')
    axes[0].set_facecolor('#2b2b2b')

    # График точности
    axes[1].plot(mfcc_history['val_acc'], label='MFCC+ (Val)', color='#4a90e2', linewidth=2, linestyle='-')
    axes[1].plot(yamnet_history['val_acc'], label='YAMNet (Val)', color='#e67e22', linewidth=2, linestyle='-')
    axes[1].plot(mfcc_history['train_acc'], label='MFCC+ (Train)', color='#4a90e2', linewidth=1, linestyle='--',
                 alpha=0.7)
    axes[1].plot(yamnet_history['train_acc'], label='YAMNet (Train)', color='#e67e22', linewidth=1, linestyle='--',
                 alpha=0.7)
    axes[1].set_xlabel('Epoch', color='#cccccc')
    axes[1].set_ylabel('Accuracy (%)', color='#cccccc')
    axes[1].set_title('Сравнение точности', color='#cccccc')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(colors='#cccccc')
    axes[1].set_facecolor('#2b2b2b')

    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
        except Exception as e:
            print(f"Ошибка сохранения графика: {e}")
    plt.close(fig)
    return fig


def plot_metrics_comparison(mfcc_metrics, yamnet_metrics, save_path=None):
    """Сравнение финальных метрик"""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    mfcc_values = [mfcc_metrics['accuracy'], mfcc_metrics['precision'], mfcc_metrics['recall'], mfcc_metrics['f1']]
    yamnet_values = [yamnet_metrics['accuracy'], yamnet_metrics['precision'], yamnet_metrics['recall'],
                     yamnet_metrics['f1']]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e1e')
    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, mfcc_values, width, label='MFCC+', color='#4a90e2', alpha=0.8)
    bars2 = ax.bar(x + width / 2, yamnet_values, width, label='YAMNet', color='#e67e22', alpha=0.8)

    ax.set_ylabel('Score', color='#cccccc')
    ax.set_title('Сравнение метрик моделей', color='#cccccc')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color='#cccccc')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(colors='#cccccc')
    ax.set_facecolor('#2b2b2b')

    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=10, color='#cccccc')

    plt.tight_layout()
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1e1e1e')
        except Exception as e:
            print(f"Ошибка сохранения графика: {e}")
    plt.close(fig)
    return fig