import torch
import cv2
import numpy as np
import argparse
import os
import yaml
import json
import glob
import shutil
from tqdm import tqdm
import matplotlib
# Используем Agg бэкенд, который не требует графического интерфейса
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from model import MultiHeadKeypointModel
from dataset import all_keypoint_classes, image_size, original_image_size, extract_points_from_json

def load_model(checkpoint_path, num_keypoints, device):
    """Загрузка обученной модели из чекпоинта"""
    model = MultiHeadKeypointModel(num_keypoints)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    """Предобработка изображения для модели"""
    # Загружаем изображение
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Сохраняем оригинальное изображение для визуализации
    orig_image = image.copy()
    
    # Изменяем размер изображения
    image = cv2.resize(image, (image_size[1], image_size[0]))
    
    # Нормализуем изображение
    image = image.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    
    # Преобразуем в тензор
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    
    return image, orig_image

def detect_keypoints(model, image, threshold=0.5):
    """Обнаружение ключевых точек на изображении"""
    with torch.no_grad():
        outputs = model(image)
    
    # Получаем предсказания
    pred_presence = torch.sigmoid(outputs[0, :, 0]).cpu().numpy()
    pred_coords = outputs[0, :, 1:].cpu().numpy()
    
    # Формируем результаты
    results = []
    for i, (presence, coords) in enumerate(zip(pred_presence, pred_coords)):
        # Денормализуем координаты обратно в пиксели
        x = int(coords[0] * original_image_size[1])
        y = int(coords[1] * original_image_size[0])
        results.append((i, all_keypoint_classes[i], x, y, presence > threshold, presence))
    
    return results

def get_gt_keypoints(json_path):
    """Получение ground truth ключевых точек из JSON-файла"""
    keypoints = extract_points_from_json(json_path)
    
    results = []
    for i, kp in enumerate(keypoints):
        if kp[0] > 0:  # Если точка присутствует
            # Денормализуем координаты обратно в пиксели
            x = int(kp[1] * original_image_size[1])
            y = int(kp[2] * original_image_size[0])
            results.append((i, all_keypoint_classes[i], x, y, True, 1.0))
        else:
            results.append((i, all_keypoint_classes[i], 0, 0, False, 0.0))
    
    return results

def visualize_comparison(image, gt_keypoints, pred_keypoints, output_path):
    """Визуализация сравнения ground truth и предсказанных точек"""
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    
    # Цвета для разных групп точек
    colors = {
        'group_1': 'red',       # CP
        'group_2': 'green',     # FE2_o, FE1_o, FE2, FE1, EC1, EC2
        'group_3': 'blue'       # CT1, CT2, CD
    }
    
    # Рисуем ground truth точки (незаполненные кружочки)
    for idx, class_name, x, y, present, _ in gt_keypoints:
        if present:
            if class_name in ['CP']:
                color = colors['group_1']
            elif class_name in ['FE2_o', 'FE1_o', 'FE2', 'FE1', 'EC1', 'EC2']:
                color = colors['group_2']
            else:
                color = colors['group_3']
            
            # Незаполненный кружочек для GT
            plt.plot(x, y, 'o', markersize=12, markerfacecolor='none', color=color, alpha=0.8, markeredgewidth=1.5)
            plt.text(x + 15, y, f"{class_name} (GT)", fontsize=12, color=color, alpha=0.7)
    
    # Рисуем предсказанные точки (жирные крестики)
    for idx, class_name, x, y, present, prob in pred_keypoints:
        if present:
            if class_name in ['CP']:
                color = colors['group_1']
            elif class_name in ['FE2_o', 'FE1_o', 'FE2', 'FE1', 'EC1', 'EC2']:
                color = colors['group_2']
            else:
                color = colors['group_3']
            
            # Жирный крестик для предсказаний
            plt.plot(x, y, 'X', markersize=10, color=color, markeredgewidth=2, alpha=0.9)
            plt.text(x + 15, y + 15, f"{class_name} ({prob:.2f})", fontsize=12, color=color)
    
    plt.title("Сравнение Ground Truth (незаполненные кружки) и предсказанных (жирные крестики) точек")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()

def calculate_metrics(gt_keypoints, pred_keypoints, distance_threshold=20):
    """Расчет метрик качества обнаружения ключевых точек"""
    # Метрики для классификации (наличие/отсутствие точки)
    y_true_presence = [kp[4] for kp in gt_keypoints]
    y_pred_presence = [kp[4] for kp in pred_keypoints]
    
    precision = precision_score(y_true_presence, y_pred_presence, zero_division=0)
    recall = recall_score(y_true_presence, y_pred_presence, zero_division=0)
    f1 = f1_score(y_true_presence, y_pred_presence, zero_division=0)
    
    # Метрики для регрессии (точность координат)
    total_points = 0
    correct_points = 0
    total_distance = 0
    
    for gt, pred in zip(gt_keypoints, pred_keypoints):
        if gt[4] and pred[4]:  # Если точка присутствует и в GT, и в предсказании
            gt_x, gt_y = gt[2], gt[3]
            pred_x, pred_y = pred[2], pred[3]
            
            # Евклидово расстояние между точками
            distance = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
            total_distance += distance
            
            # Считаем точку корректной, если расстояние меньше порога
            if distance < distance_threshold:
                correct_points += 1
            
            total_points += 1
    
    # Средняя ошибка расстояния и точность локализации
    avg_distance_error = total_distance / total_points if total_points > 0 else float('inf')
    localization_accuracy = correct_points / total_points if total_points > 0 else 0
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'avg_distance_error': avg_distance_error,
        'localization_accuracy': localization_accuracy,
        'total_points': total_points,
        'correct_points': correct_points
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Тестирование модели обнаружения ключевых точек')
    parser.add_argument('--config', type=str, default='config.yaml', help='Путь к конфигурационному файлу')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Путь к чекпоинту модели')
    parser.add_argument('--folder', type=str, default='/home/knru/80lab/Gergent/TAVI_new/321814_Tavi_detection/0001/', 
                        help='Путь к папке с тестовыми изображениями')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Папка для сохранения результатов')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог вероятности для фильтрации точек')
    parser.add_argument('--distance_threshold', type=float, default=20, 
                        help='Порог расстояния для определения корректности локализации (в пикселях)')
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Загружаем модель
    model = load_model(args.checkpoint, len(all_keypoint_classes), device)
    print(f"Модель загружена из {args.checkpoint}")
    
    # Создаем директорию для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Получаем список всех JSON-файлов с аннотациями
    json_files = sorted(glob.glob(os.path.join(args.folder, 'ann', '*.png.json')))
    print(f"Найдено {len(json_files)} файлов с аннотациями")
    
    # Для хранения общих метрик
    all_metrics = []
    
    # Обрабатываем каждое изображение
    for json_file in tqdm(json_files, desc="Обработка изображений"):
        # Получаем путь к изображению
        img_name = os.path.basename(json_file).replace('.json', '')
        img_path = os.path.join(args.folder, 'img', img_name)
        
        if not os.path.exists(img_path):
            print(f"Изображение не найдено: {img_path}")
            continue
        
        # Предобрабатываем изображение
        image_tensor, orig_image = preprocess_image(img_path)
        image_tensor = image_tensor.to(device)
        
        # Получаем ground truth ключевые точки
        gt_keypoints = get_gt_keypoints(json_file)
        
        # Обнаруживаем ключевые точки
        pred_keypoints = detect_keypoints(model, image_tensor, args.threshold)
        
        # Рассчитываем метрики
        metrics = calculate_metrics(gt_keypoints, pred_keypoints, args.distance_threshold)
        metrics['image'] = img_name
        all_metrics.append(metrics)
        
        # Визуализируем результаты
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(img_name)[0]}_comparison.png")
        visualize_comparison(orig_image, gt_keypoints, pred_keypoints, output_path)
    
    # Рассчитываем средние метрики
    avg_metrics = {
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        'avg_distance_error': np.mean([m['avg_distance_error'] for m in all_metrics if m['avg_distance_error'] != float('inf')]),
        'localization_accuracy': np.mean([m['localization_accuracy'] for m in all_metrics]),
        'total_images': len(all_metrics)
    }
    
    # Сохраняем метрики в JSON
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({'per_image': all_metrics, 'average': avg_metrics}, f, indent=4)
    
    # Выводим средние метрики
    print("\nСредние метрики:")
    print(f"Precision: {avg_metrics['precision']:.4f}")
    print(f"Recall: {avg_metrics['recall']:.4f}")
    print(f"F1 Score: {avg_metrics['f1_score']:.4f}")
    print(f"Средняя ошибка расстояния: {avg_metrics['avg_distance_error']:.2f} пикселей")
    print(f"Точность локализации: {avg_metrics['localization_accuracy']:.4f}")
    print(f"Всего изображений: {avg_metrics['total_images']}")
    print(f"\nРезультаты сохранены в папке: {args.output_dir}")

if __name__ == '__main__':
    main()
