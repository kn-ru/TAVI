import os
import yaml
import torch
import torch.nn as nn
import numpy as np
import json
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Используем Agg бэкенд для работы без GUI
from tqdm import tqdm

from dataset import TAVIDataset, all_keypoint_classes
from heatmap_model import HeatmapKeypointModel

def calculate_metrics(pred_keypoints, gt_keypoints, distance_threshold=0.05, image_size=(512, 512)):
    """
    Вычисляет метрики для оценки качества предсказания ключевых точек.
    
    Args:
        pred_keypoints: Предсказанные координаты [batch_size, num_keypoints, 3]
        gt_keypoints: Ground truth координаты [batch_size, num_keypoints, 3]
        distance_threshold: Порог расстояния для определения правильного предсказания
        
    Returns:
        dict: Словарь с метриками
    """
    batch_size = pred_keypoints.size(0)
    num_keypoints = pred_keypoints.size(1)
    
    # Инициализируем счетчики
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_distance_error = 0  # в нормализованных координатах
    total_distance_error_px = 0  # в пикселях
    total_points = 0
    points_within_threshold = 0
    points_within_px_threshold = 0
    
    # Порог в пикселях (например, 5 пикселей)
    px_threshold = 5.0
    
    # Вычисляем метрики для каждой точки
    for b in range(batch_size):
        for k in range(num_keypoints):
            gt_present = gt_keypoints[b, k, 0] > 0
            pred_present = pred_keypoints[b, k, 0] > 0
            
            if gt_present and pred_present:
                # Вычисляем расстояние между предсказанной и ground truth точками
                gt_x, gt_y = gt_keypoints[b, k, 1], gt_keypoints[b, k, 2]
                pred_x, pred_y = pred_keypoints[b, k, 1], pred_keypoints[b, k, 2]
                
                # Расстояние в нормализованных координатах [0,1]
                distance = torch.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
                
                # Расстояние в пикселях
                gt_x_px = gt_x * image_size[1]  # width
                gt_y_px = gt_y * image_size[0]  # height
                pred_x_px = pred_x * image_size[1]
                pred_y_px = pred_y * image_size[0]
                distance_px = torch.sqrt((gt_x_px - pred_x_px)**2 + (gt_y_px - pred_y_px)**2)
                
                total_distance_error += distance.item()
                total_distance_error_px += distance_px.item()
                total_points += 1
                
                if distance < distance_threshold:
                    points_within_threshold += 1
                    
                if distance_px < px_threshold:
                    points_within_px_threshold += 1
                
                true_positives += 1
            elif gt_present and not pred_present:
                false_negatives += 1
            elif not gt_present and pred_present:
                false_positives += 1
    
    # Вычисляем метрики
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    avg_distance_error = total_distance_error / total_points if total_points > 0 else 0
    avg_distance_error_px = total_distance_error_px / total_points if total_points > 0 else 0
    localization_accuracy = points_within_threshold / total_points if total_points > 0 else 0
    localization_accuracy_px = points_within_px_threshold / total_points if total_points > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_distance_error': avg_distance_error,
        'avg_distance_error_px': avg_distance_error_px,
        'localization_accuracy': localization_accuracy,
        'localization_accuracy_px': localization_accuracy_px,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_points': total_points,
        'points_within_threshold': points_within_threshold,
        'points_within_px_threshold': points_within_px_threshold
    }

def test(config):
    """
    Функция для тестирования модели с тепловыми картами.
    
    Args:
        config: Словарь с параметрами конфигурации
    """
    # Инициализируем тестовый датасет
    test_dataset = TAVIDataset(config['dataset_path'], mode='val')  # Используем валидационный набор для тестирования
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Инициализируем модель
    heatmap_size = config.get('heatmap_size', (64, 64))
    model = HeatmapKeypointModel(
        backbone_type='hrnet',
        heatmap_size=heatmap_size,
        sigma=config.get('gaussian_sigma', 2.0)
    )
    
    # Загружаем веса модели
    checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_heatmap_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
    else:
        print(f"Checkpoint not found at {checkpoint_path}. Using randomly initialized model.")
    
    # Определяем устройство (GPU или CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Переводим модель в режим оценки
    model.eval()
    
    # Инициализируем переменные для хранения метрик
    all_metrics = []
    
    # Создаем директорию для сохранения результатов
    results_dir = os.path.join(config.get('results_dir', 'results'), 'heatmap')
    os.makedirs(results_dir, exist_ok=True)
    
    # Проходим по тестовому датасету
    with torch.no_grad():
        for batch_idx, (images, keypoints, img_paths) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            # Получаем предсказания модели
            outputs = model(images)
            pred_keypoints = outputs['keypoints']
            pred_heatmaps = outputs['heatmaps']
            
            # Вычисляем метрики
            batch_metrics = calculate_metrics(pred_keypoints, keypoints)
            all_metrics.append(batch_metrics)
            
            # Сохраняем визуализацию для первых нескольких батчей
            if batch_idx < config.get('visualization', {}).get('max_batches', 5):
                vis_path = os.path.join(results_dir, f'batch_{batch_idx}_heatmap.png')
                
                # Визуализируем тепловые карты и предсказанные точки
                visualize_heatmaps_and_keypoints(
                    images.cpu(),
                    keypoints.cpu(),
                    pred_keypoints.cpu(),
                    pred_heatmaps.cpu(),
                    model.generate_target_heatmaps(keypoints.cpu(), heatmap_size),
                    save_path=vis_path,
                    max_images=config.get('visualization', {}).get('max_images', 4)
                )
    
    # Вычисляем средние метрики
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        'avg_distance_error': np.mean([m['avg_distance_error'] for m in all_metrics]),
        'avg_distance_error_px': np.mean([m['avg_distance_error_px'] for m in all_metrics]),
        'localization_accuracy': np.mean([m['localization_accuracy'] for m in all_metrics]),
        'localization_accuracy_px': np.mean([m['localization_accuracy_px'] for m in all_metrics]),
        'true_positives': sum([m['true_positives'] for m in all_metrics]),
        'false_positives': sum([m['false_positives'] for m in all_metrics]),
        'false_negatives': sum([m['false_negatives'] for m in all_metrics]),
        'total_points': sum([m['total_points'] for m in all_metrics]),
        'points_within_threshold': sum([m['points_within_threshold'] for m in all_metrics]),
        'points_within_px_threshold': sum([m['points_within_px_threshold'] for m in all_metrics])
    }
    
    # Выводим результаты
    print("\nTest Results:")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F1 Score: {avg_metrics['f1_score']:.4f}")
    print(f"Average Distance Error: {avg_metrics['avg_distance_error']:.4f} (normalized)")
    print(f"Average Distance Error: {avg_metrics['avg_distance_error_px']:.2f} pixels")
    print(f"Localization Accuracy: {avg_metrics['localization_accuracy']:.4f} (threshold: 0.05)")
    print(f"Localization Accuracy: {avg_metrics['localization_accuracy_px']:.4f} (threshold: 5 pixels)")
    print(f"True Positives: {avg_metrics['true_positives']}")
    print(f"False Positives: {avg_metrics['false_positives']}")
    print(f"False Negatives: {avg_metrics['false_negatives']}")
    
    # Сохраняем метрики в JSON файл
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    
    print(f"\nMetrics saved to {metrics_path}")
    
    return avg_metrics

def visualize_heatmaps_and_keypoints(images, gt_keypoints, pred_keypoints, pred_heatmaps, target_heatmaps, save_path, max_images=4):
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
    """
    batch_size = min(images.size(0), max_images)
    num_keypoints = gt_keypoints.size(1)
    
    # Создаем фигуру
    fig, axes = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))
    
    # Если batch_size = 1, то axes не будет массивом, делаем его массивом
    if batch_size == 1:
        axes = np.array([axes])
    
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
                x, y = gt_keypoints[i, k, 1] * images[i].size(2), gt_keypoints[i, k, 2] * images[i].size(1)
                axes[i, 0].plot(x, y, 'ro', markersize=5)
        axes[i, 0].set_title('Ground Truth')
        axes[i, 0].axis('off')
        
        # Изображение с предсказанными точками
        axes[i, 1].imshow(images_np[i])
        for k in range(num_keypoints):
            if pred_keypoints[i, k, 0] > 0:  # Если точка присутствует
                x, y = pred_keypoints[i, k, 1] * images[i].size(2), pred_keypoints[i, k, 2] * images[i].size(1)
                axes[i, 1].plot(x, y, 'go', markersize=5)
        axes[i, 1].set_title('Predictions')
        axes[i, 1].axis('off')
        
        # Суммарная тепловая карта (сумма по всем каналам)
        sum_heatmap = torch.sum(pred_heatmaps[i], dim=0).numpy()
        axes[i, 2].imshow(images_np[i])
        axes[i, 2].imshow(sum_heatmap, alpha=0.6, cmap='jet')
        axes[i, 2].set_title('Predicted Heatmap')
        axes[i, 2].axis('off')
    
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
    if 'results_dir' not in config:
        config['results_dir'] = 'results'
    
    test(config)
