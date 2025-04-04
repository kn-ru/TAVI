import os
import yaml
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # Используем Agg бэкенд для работы без GUI
import matplotlib.pyplot as plt
import numpy as np
import cv2

from dataset import TAVIDataset, all_keypoint_classes
from heatmap_model import HeatmapKeypointModel
from visualization import create_batch_visualization

class HeatmapLoss(nn.Module):
    """
    Функция потерь для обучения модели с тепловыми картами.
    Включает MSE Loss для тепловых карт и опционально Focal Loss для улучшения работы с несбалансированными данными.
    """
    def __init__(self, use_focal_loss=True, alpha=2.0, beta=4.0):
        super(HeatmapLoss, self).__init__()
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha  # Параметр для focal loss
        self.beta = beta    # Параметр для focal loss
        
    def forward(self, pred_heatmaps, target_heatmaps):
        """
        Вычисление функции потерь.
        
        Args:
            pred_heatmaps: Предсказанные тепловые карты [batch, num_keypoints, H, W]
            target_heatmaps: Целевые тепловые карты [batch, num_keypoints, H, W]
            
        Returns:
            dict: Словарь с компонентами потерь
        """
        batch_size = pred_heatmaps.size(0)
        
        # Вычисляем потерю
        
        # Используем простой MSE Loss для начала
        mse_loss = F.mse_loss(pred_heatmaps, target_heatmaps)
        
        if self.use_focal_loss:
            # Focal MSE Loss для тепловых карт
            # Формула: FL = ((1 - p)^alpha * (p)^beta * (y - p)^2) для y=1
            #          FL = ((1 - p)^beta * (p)^alpha * (y - p)^2) для y=0
            # где p - предсказанное значение, y - целевое значение
            
            # Используем порог для определения положительных пикселей
            # В гауссовом распределении максимальное значение обычно около 0.6-0.7
            pos_threshold = 0.1  # Уменьшаем порог, чтобы захватить больше положительных пикселей
            pos_mask = (target_heatmaps > pos_threshold).float()
            neg_mask = (target_heatmaps <= pos_threshold).float()
            
            # Подсчитываем количество положительных и отрицательных пикселей
            pos_pixels = pos_mask.sum().item()
            neg_pixels = neg_mask.sum().item()
            total_pixels = pos_mask.numel()
            
            pos_weights = torch.pow(1 - pred_heatmaps, self.alpha) * torch.pow(pred_heatmaps, self.beta)
            neg_weights = torch.pow(1 - pred_heatmaps, self.beta) * torch.pow(pred_heatmaps, self.alpha)
            
            pos_loss = pos_mask * pos_weights * (target_heatmaps - pred_heatmaps) ** 2
            neg_loss = neg_mask * neg_weights * (target_heatmaps - pred_heatmaps) ** 2
            
            # Вычисляем значения потерь
            pos_loss_val = pos_loss.sum().item() / (pos_pixels + 1e-6)
            neg_loss_val = neg_loss.sum().item() / (neg_pixels + 1e-6)
            
            # Используем простой MSE вместо Focal Loss для отладки
            # heatmap_loss = (pos_loss + neg_loss).mean()
            heatmap_loss = mse_loss  # Временно используем MSE для отладки
        else:
            # Обычный MSE Loss
            heatmap_loss = mse_loss
        
        return {
            'total': heatmap_loss,
            'heatmap': heatmap_loss
        }

def train(config):
    """
    Функция для обучения модели с тепловыми картами.
    
    Args:
        config: Словарь с параметрами конфигурации
    """
    # Создаем директории для сохранения результатов
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Создаем директорию для визуализаций, если она включена
    if config.get('visualization', {}).get('save_batch_images', False):
        vis_dir = config.get('visualization', {}).get('output_dir', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Инициализируем датасеты и загрузчики данных
    train_dataset = TAVIDataset(config['dataset_path'], mode='train')
    val_dataset = TAVIDataset(config['dataset_path'], mode='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Инициализируем модель
    heatmap_size = config.get('heatmap_size', (64, 64))
    model = HeatmapKeypointModel(
        backbone_type='hrnet',
        heatmap_size=heatmap_size,
        sigma=config.get('gaussian_sigma', 2.0)
    )
    
    # Определяем устройство (GPU или CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    model.to(device)
    
    # Инициализируем оптимизатор и планировщик скорости обучения
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Инициализируем функцию потерь
    criterion = HeatmapLoss(
        use_focal_loss=config.get('use_focal_loss', True),
        alpha=config.get('focal_alpha', 2.0),
        beta=config.get('focal_beta', 4.0)
    )
    
    # Инициализируем TensorBoard для логирования
    writer = SummaryWriter(f"runs/heatmap_{time.strftime('%Y%m%d-%H%M%S')}")
    
    # Лучшая потеря на валидации для сохранения модели
    best_val_loss = float('inf')
    
    # Основной цикл обучения
    for epoch in range(1, config['epochs'] + 1):
        epoch_start_time = time.time()
        
        # Обучение
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Теперь batch_data содержит три элемента: изображения, ключевые точки и пути к файлам
            images, keypoints, file_paths = batch_data
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            # Пропускаем вывод отладочной информации
            
            # Генерируем целевые тепловые карты из координат ключевых точек
            target_heatmaps = model.generate_target_heatmaps(keypoints, heatmap_size)
            
            optimizer.zero_grad()
            
            # Прямой проход
            outputs = model(images)
            pred_heatmaps = outputs['heatmaps']
            
            # Пропускаем вывод отладочной информации
            
            # Изменяем размер целевых тепловых карт, чтобы он совпадал с предсказанными
            if pred_heatmaps.shape != target_heatmaps.shape:
                target_heatmaps = F.interpolate(
                    target_heatmaps,
                    size=(pred_heatmaps.shape[2], pred_heatmaps.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
                # Размер целевых тепловых карт изменен
            
            # Вычисляем потерю
            loss_dict = criterion(pred_heatmaps, target_heatmaps)
            loss = loss_dict['total']
            
            # Обратный проход и оптимизация
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Выводим информацию о процессе обучения реже
            if (batch_idx + 1) % 100 == 0:
                print(f"Эпоха {epoch}/{config['epochs']} [{batch_idx+1}/{len(train_loader)}] "
                      f"Потеря: {loss.item():.6f}")
            
            # Сохраняем визуализацию первого батча в каждой эпохе
            if batch_idx == 0 and config.get('visualization', {}).get('save_batch_images', False):
                vis_dir = config.get('visualization', {}).get('output_dir', 'visualizations')
                vis_path = os.path.join(vis_dir, f'epoch_{epoch}_heatmap.png')
                
                # Используем пути к файлам, которые возвращает DataLoader
                # Берем первые max_images путей из батча
                max_images = config.get('visualization', {}).get('max_images', 4)
                batch_file_paths = file_paths[:max_images]
                
                # Визуализируем тепловые карты и предсказанные точки
                visualize_heatmaps_and_keypoints(
                    images.cpu(),
                    keypoints.cpu(),
                    outputs['keypoints'].detach().cpu(),
                    pred_heatmaps.detach().cpu(),
                    target_heatmaps.cpu(),
                    save_path=vis_path,
                    max_images=max_images,
                    file_paths=batch_file_paths
                )
                
                # Загружаем сохраненное изображение и добавляем его в TensorBoard
                img = plt.imread(vis_path)
                writer.add_image('Heatmap Visualization', img.transpose(2, 0, 1), epoch)
        
        # Вычисляем среднюю потерю за эпоху
        train_loss /= len(train_loader)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Теперь batch_data содержит три элемента: изображения, ключевые точки и пути к файлам
                images, keypoints, _ = batch_data
                images = images.to(device)
                keypoints = keypoints.to(device)
                
                # Генерируем целевые тепловые карты
                target_heatmaps = model.generate_target_heatmaps(keypoints, heatmap_size)
                
                # Прямой проход
                outputs = model(images)
                pred_heatmaps = outputs['heatmaps']
                
                # Изменяем размер целевых тепловых карт, чтобы он совпадал с предсказанными
                if pred_heatmaps.shape != target_heatmaps.shape:
                    target_heatmaps = F.interpolate(
                        target_heatmaps,
                        size=(pred_heatmaps.shape[2], pred_heatmaps.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                
                # Вычисляем потерю
                loss_dict = criterion(pred_heatmaps, target_heatmaps)
                val_loss += loss_dict['total'].item()
        
        # Вычисляем среднюю потерю за эпоху
        val_loss /= len(val_loader)
        
        # Вычисляем время выполнения эпохи
        epoch_time = time.time() - epoch_start_time
        
        # Обновляем планировщик скорости обучения
        scheduler.step(val_loss)
        
        # Логируем метрики в TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Выводим краткую информацию о результатах эпохи
        print(f"Эпоха {epoch}/{config['epochs']} | Потеря: {train_loss:.4f}/{val_loss:.4f} | Время: {epoch_time:.1f}с")
        
        # Сохраняем модель, если она лучше предыдущих
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_heatmap_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"Сохранена лучшая модель")
        
        # Сохраняем последнюю модель
        if epoch % 10 == 0 or epoch == config['epochs']:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'heatmap_model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
    
    print("Обучение завершено!")
    writer.close()

def visualize_heatmaps_and_keypoints(images, gt_keypoints, pred_keypoints, pred_heatmaps, target_heatmaps, save_path, max_images=4, file_paths=None):
    """
    Визуализирует изображения, тепловые карты и ключевые точки.
    
    Args:
        images: Тензор с изображениями [batch_size, 3, H, W]
        gt_keypoints: Тензор с ground truth точками [batch_size, num_keypoints, 3]
        pred_keypoints: Тензор с предсказанными точками [batch_size, num_keypoints, 3]
        pred_heatmaps: Тензор с предсказанными тепловыми картами [batch_size, num_keypoints, H, W]
        target_heatmaps: Тензор с целевыми тепловыми картами [batch_size, num_keypoints, H, W]
        save_path: Путь для сохранения визуализации
        max_images: Максимальное количество изображений для визуализации
        debug_info: Дополнительная информация для отладки (пути к файлам и т.д.)
    """
    """
    Визуализирует изображения, тепловые карты и ключевые точки.
    
    Args:
        images: Тензор с изображениями [batch_size, 3, H, W]
        gt_keypoints: Тензор с ground truth точками [batch_size, num_keypoints, 3]
        pred_keypoints: Тензор с предсказанными точками [batch_size, num_keypoints, 3]
        pred_heatmaps: Тензор с предсказанными тепловыми картами [batch_size, num_keypoints, H, W]
        target_heatmaps: Тензор с целевыми тепловыми картами [batch_size, num_keypoints, H, W]
        save_path: Путь для сохранения визуализации
        max_images: Максимальное количество изображений для визуализации
        debug_info: Дополнительная информация для отладки (пути к файлам и т.д.)
    """
    batch_size = min(images.size(0), max_images)
    num_keypoints = gt_keypoints.size(1)
    
    # Создаем фигуру
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    
    # Если batch_size = 1, то axes не будет массивом, делаем его массивом
    if batch_size == 1:
        axes = np.array([axes])
        
    # Добавляем заголовок
    plt.suptitle(f"Heatmap Keypoint Detection", fontsize=16)
    
    # Денормализуем изображения
    images_np = []
    for i in range(batch_size):
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        images_np.append(img)
    
    # Визуализируем каждое изображение и соответствующие тепловые карты
    for i in range(batch_size):
        # Изображение с ground truth точками
        axes[i, 0].imshow(images_np[i])
        for k in range(num_keypoints):
            if gt_keypoints[i, k, 0] > 0:  # Если точка присутствует
                # Просто умножаем нормализованные координаты на размер изображения
                x = gt_keypoints[i, k, 1] * images[i].size(2)
                y = gt_keypoints[i, k, 2] * images[i].size(1)
                
                # Добавляем номер точки для отладки
                axes[i, 0].plot(x, y, 'ro', markersize=5)
                axes[i, 0].text(x+5, y+5, f"{k}", color='red', fontsize=8)
        axes[i, 0].set_title('Ground Truth')
        
        # Добавляем путь к файлу на изображение
        if file_paths and i < len(file_paths):
            # Добавляем текст внизу изображения
            img_h = images[i].size(1)
            axes[i, 0].text(5, img_h - 10, file_paths[i], color='white', fontsize=10, 
                           bbox=dict(facecolor='black', alpha=0.7))
        axes[i, 0].axis('off')
        
        # Изображение с предсказанными точками
        axes[i, 1].imshow(images_np[i])
        for k in range(num_keypoints):
            if pred_keypoints[i, k, 0] > 0:  # Если точка присутствует
                # Просто умножаем нормализованные координаты на размер изображения
                x = pred_keypoints[i, k, 1] * images[i].size(2)
                y = pred_keypoints[i, k, 2] * images[i].size(1)
                
                axes[i, 1].plot(x, y, 'go', markersize=5)
                # Добавляем номер точки для отладки
                axes[i, 1].text(x+5, y+5, f"{k}", color='green', fontsize=8)
        axes[i, 1].set_title('Predictions')
        
        # Добавляем путь к файлу на изображение
        if file_paths and i < len(file_paths):
            # Добавляем текст внизу изображения
            img_h = images[i].size(1)
            axes[i, 1].text(5, img_h - 10, file_paths[i], color='white', fontsize=10, 
                           bbox=dict(facecolor='black', alpha=0.7))
        axes[i, 1].axis('off')
        
        # Суммарная тепловая карта (сумма по всем каналам)
        sum_heatmap = torch.sum(pred_heatmaps[i], dim=0).numpy()
        
        # Получаем размеры изображения и тепловой карты
        img_h, img_w = images_np[i].shape[:2]
        heatmap_h, heatmap_w = sum_heatmap.shape
        
        # Масштабируем тепловую карту до размера изображения
        resized_heatmap = cv2.resize(sum_heatmap, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        
        # Отображаем изображение и накладываем на него тепловую карту
        axes[i, 2].imshow(images_np[i])
        heatmap_img = axes[i, 2].imshow(resized_heatmap, alpha=0.6, cmap='jet')
        axes[i, 2].set_title('Predicted Heatmap')
        
        # Добавляем путь к файлу на изображение
        if file_paths and i < len(file_paths):
            # Добавляем текст внизу изображения
            img_h = images[i].size(1)
            axes[i, 2].text(5, img_h - 10, file_paths[i], color='white', fontsize=10, 
                           bbox=dict(facecolor='black', alpha=0.7))
        axes[i, 2].axis('off')
        
        # Добавляем цветовую шкалу
        if i == 0:  # Только для первого изображения
            cbar = plt.colorbar(heatmap_img, ax=axes[i, 2], fraction=0.046, pad=0.04)
            cbar.set_label('Intensity', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

if __name__ == "__main__":
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Добавляем параметры для heatmap-модели, если их нет
    if 'heatmap_size' not in config:
        config['heatmap_size'] = (64, 64)
    if 'gaussian_sigma' not in config:
        config['gaussian_sigma'] = 2.0
    if 'use_focal_loss' not in config:
        config['use_focal_loss'] = True
    if 'focal_alpha' not in config:
        config['focal_alpha'] = 2.0
    if 'focal_beta' not in config:
        config['focal_beta'] = 4.0
    
    train(config)
