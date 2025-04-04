import os
import json
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Используем Agg бэкенд для работы без GUI
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import TAVIDataset
from hrnet_model import HRNetKeypointModel
from visualization import create_batch_visualization

def calculate_metrics(pred_keypoints, gt_keypoints, distance_threshold=0.05):
    """
    Вычисляет метрики для оценки качества предсказания ключевых точек:
    - Precision, Recall, F1 для определения присутствия точек
    - Средняя ошибка расстояния между предсказанными и ground truth точками
    - Точность локализации (процент точек в пределах порогового расстояния)
    
    Args:
        pred_keypoints: Предсказанные ключевые точки [batch, num_keypoints, 3]
        gt_keypoints: Ground truth ключевые точки [batch, num_keypoints, 3]
        distance_threshold: Пороговое расстояние для определения правильной локализации
        
    Returns:
        dict: Словарь с метриками
    """
    # Извлекаем информацию о присутствии точек
    pred_presence = (pred_keypoints[:, :, 0] > 0.5).float()
    gt_presence = (gt_keypoints[:, :, 0] > 0).float()
    
    # Вычисляем метрики присутствия
    true_positives = torch.sum((pred_presence == 1) & (gt_presence == 1)).item()
    false_positives = torch.sum((pred_presence == 1) & (gt_presence == 0)).item()
    false_negatives = torch.sum((pred_presence == 0) & (gt_presence == 1)).item()
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Вычисляем ошибку расстояния только для точек, которые присутствуют в ground truth
    mask = (gt_presence == 1)
    if torch.sum(mask) == 0:
        avg_distance_error = 0.0
        localization_accuracy = 0.0
    else:
        # Вычисляем евклидово расстояние между предсказанными и ground truth координатами
        pred_coords = pred_keypoints[mask, 1:3]
        gt_coords = gt_keypoints[mask, 1:3]
        
        distances = torch.sqrt(torch.sum((pred_coords - gt_coords) ** 2, dim=1))
        avg_distance_error = torch.mean(distances).item()
        
        # Вычисляем точность локализации (процент точек в пределах порогового расстояния)
        localization_accuracy = torch.mean((distances < distance_threshold).float()).item()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_distance_error': avg_distance_error,
        'localization_accuracy': localization_accuracy
    }

def test(config):
    # Инициализируем тестовый датасет
    test_dataset = TAVIDataset(config['dataset_path'], mode='val')  # Используем валидационный набор для тестирования
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Инициализируем модель HRNet
    model = HRNetKeypointModel(width=config.get('hrnet_width', 32))
    
    # Загружаем веса модели
    checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_hrnet_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
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
    results_dir = os.path.join(config['results_dir'], 'hrnet')
    os.makedirs(results_dir, exist_ok=True)
    
    # Проходим по тестовому датасету
    with torch.no_grad():
        for batch_idx, (images, keypoints) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            keypoints = keypoints.to(device)
            
            # В TAVIDataset нет group_labels, поэтому мы его не используем
            
            # Получаем предсказания модели
            outputs = model(images)
            pred_keypoints = outputs['keypoints']
            
            # Вычисляем метрики
            batch_metrics = calculate_metrics(pred_keypoints, keypoints)
            all_metrics.append(batch_metrics)
            
            # Сохраняем визуализацию для первых нескольких батчей
            if batch_idx < config.get('visualization', {}).get('max_batches', 5):
                vis_path = os.path.join(results_dir, f'batch_{batch_idx}_hrnet.png')
                create_batch_visualization(
                    images.cpu(),
                    keypoints.cpu(),
                    pred_keypoints.cpu(),
                    save_path=vis_path,
                    max_images=config.get('visualization', {}).get('max_images', 16)
                )
    
    # Вычисляем средние метрики
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        'avg_distance_error': np.mean([m['avg_distance_error'] for m in all_metrics]),
        'localization_accuracy': np.mean([m['localization_accuracy'] for m in all_metrics])
    }
    
    # Выводим результаты
    print("\nTest Results (HRNet):")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F1 Score: {avg_metrics['f1_score']:.4f}")
    print(f"Average Distance Error: {avg_metrics['avg_distance_error']:.4f}")
    print(f"Localization Accuracy: {avg_metrics['localization_accuracy']:.4f}")
    
    # Сохраняем метрики в JSON файл
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=4)
    
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Добавляем параметры для HRNet
    config['hrnet_width'] = 32  # Ширина каналов в HRNet
    config['results_dir'] = 'results'  # Директория для сохранения результатов
    
    # Создаем директорию для результатов, если она не существует
    os.makedirs(config['results_dir'], exist_ok=True)
    
    test(config)
