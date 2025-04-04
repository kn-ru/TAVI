import torch
import numpy as np
import matplotlib
# Используем Agg бэкенд, который не требует графического интерфейса
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from dataset import all_keypoint_classes, group_1, group_2, group_3

def denormalize_image(image):
    """
    Денормализует изображение из тензора в RGB-изображение
    """
    # Клонируем тензор, чтобы не изменять оригинал
    image = image.clone().detach().cpu()
    
    # Денормализуем
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    
    # Преобразуем в numpy и транспонируем
    image = image.permute(1, 2, 0).numpy()
    
    # Обрезаем значения до [0, 1]
    image = np.clip(image, 0, 1)
    
    return image

def create_batch_visualization(images, keypoints, predictions, save_path, max_images=16):
    """
    Создает коллаж из изображений с наложенными предсказанными и ground truth ключевыми точками
    
    Args:
        images: тензор с изображениями [batch_size, 3, H, W]
        keypoints: тензор с ground truth точками [batch_size, num_keypoints, 3]
        predictions: выход модели - словарь или тензор [batch_size, num_keypoints, 3]
        save_path: путь для сохранения коллажа
        max_images: максимальное количество изображений в коллаже
    """
    # Получаем размер батча
    batch_size = images.shape[0]
    
    # Ограничиваем количество изображений
    n_images = min(batch_size, max_images)
    
    # Определяем размер сетки
    grid_size = int(np.ceil(np.sqrt(n_images)))
    
    # Создаем фигуру
    fig = plt.figure(figsize=(4*grid_size, 4*grid_size))
    
    # Обрабатываем предсказания
    if isinstance(predictions, dict):
        pred_keypoints = predictions['keypoints']
    else:
        pred_keypoints = predictions
    
    # Преобразуем вероятности присутствия в sigmoid
    pred_presence = torch.sigmoid(pred_keypoints[:, :, 0])
    
    # Цвета для разных групп точек
    colors = {
        'group_1': 'red',       # CP
        'group_2': 'green',     # FE2_o, FE1_o, FE2, FE1, EC1, EC2
        'group_3': 'blue'       # CT1, CT2, CD
    }
    
    # Создаем индексы для каждой группы
    group1_indices = [all_keypoint_classes.index(cls) for cls in group_1]
    group2_indices = [all_keypoint_classes.index(cls) for cls in group_2]
    group3_indices = [all_keypoint_classes.index(cls) for cls in group_3]
    
    for i in range(n_images):
        # Добавляем подграфик
        ax = fig.add_subplot(grid_size, grid_size, i+1)
        
        # Денормализуем изображение
        img = denormalize_image(images[i])
        
        # Отображаем изображение
        ax.imshow(img)
        
        # Отображаем ground truth точки (сплошные круги)
        for j in range(keypoints.shape[1]):
            # Если точка присутствует
            if keypoints[i, j, 0] > 0.5:
                # Определяем цвет в зависимости от группы
                if j in group1_indices:
                    color = colors['group_1']
                elif j in group2_indices:
                    color = colors['group_2']
                else:
                    color = colors['group_3']
                
                # Координаты точки (денормализованные)
                x = keypoints[i, j, 1].item() * img.shape[1]
                y = keypoints[i, j, 2].item() * img.shape[0]
                
                # Рисуем точку
                ax.plot(x, y, 'o', markersize=8, color=color, alpha=0.7)
        
        # Отображаем предсказанные точки (круги с крестиком)
        for j in range(pred_keypoints.shape[1]):
            # Если вероятность присутствия выше порога
            if pred_presence[i, j] > 0.5:
                # Определяем цвет в зависимости от группы
                if j in group1_indices:
                    color = colors['group_1']
                elif j in group2_indices:
                    color = colors['group_2']
                else:
                    color = colors['group_3']
                
                # Координаты точки (денормализованные)
                x = pred_keypoints[i, j, 1].item() * img.shape[1]
                y = pred_keypoints[i, j, 2].item() * img.shape[0]
                
                # Рисуем точку
                ax.plot(x, y, 'x', markersize=6, color=color)
                ax.plot(x, y, 'o', markersize=8, markerfacecolor='none', color=color)
        
        # Убираем оси
        ax.axis('off')
    
    # Сохраняем фигуру
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close(fig)
    
    return save_path
