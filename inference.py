import torch
import cv2
import numpy as np
import argparse
import os
import yaml
# Устанавливаем бэкенд matplotlib на Agg для работы без GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Импортируем обе модели
from model import MultiHeadKeypointModel
from heatmap_model import HeatmapKeypointModel
from dataset import all_keypoint_classes, image_size, original_image_size

def load_model(checkpoint_path, model_type='regression', device='cpu'):
    """Загрузка обученной модели из чекпоинта
    
    Args:
        checkpoint_path: Путь к файлу чекпоинта
        model_type: Тип модели ('regression' или 'heatmap')
        device: Устройство для загрузки модели
    
    Returns:
        Загруженная модель
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_type == 'regression':
        # Загрузка модели с мульти-головой регрессией
        model = MultiHeadKeypointModel(len(all_keypoint_classes))
    elif model_type == 'heatmap':
        # Загрузка модели с тепловыми картами
        model = HeatmapKeypointModel(backbone_type='hrnet', heatmap_size=(64, 64), sigma=2.0)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}. Допустимые значения: 'regression' или 'heatmap'")
    
    # Загружаем веса модели
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

def detect_keypoints(model, image, threshold=0.5, model_type='regression'):
    """Обнаружение ключевых точек на изображении
    
    Args:
        model: Загруженная модель
        image: Тензор изображения
        threshold: Порог вероятности для фильтрации точек
        model_type: Тип модели ('regression' или 'heatmap')
        
    Returns:
        Список обнаруженных ключевых точек
    """
    with torch.no_grad():
        outputs = model(image)
    
    detected_keypoints = []
    
    if model_type == 'regression':
        # Обработка выхода модели с мульти-головой регрессией
        pred_presence = torch.sigmoid(outputs[0, :, 0]).cpu().numpy()
        pred_coords = outputs[0, :, 1:].cpu().numpy()
        
        # Фильтруем точки по порогу вероятности
        for i, (presence, coords) in enumerate(zip(pred_presence, pred_coords)):
            if presence > threshold:
                # Денормализуем координаты обратно в пиксели
                x = int(coords[0] * original_image_size[1])
                y = int(coords[1] * original_image_size[0])
                detected_keypoints.append((i, all_keypoint_classes[i], x, y, presence))
    
    elif model_type == 'heatmap':
        # Обработка выхода модели с тепловыми картами
        # В этом случае outputs - это словарь с ключами 'heatmaps' и 'keypoints'
        pred_heatmaps = outputs['heatmaps'][0].cpu().numpy()  # [num_keypoints, H, W]
        pred_keypoints = outputs['keypoints'][0].cpu().numpy()  # [num_keypoints, 3] - [presence, x, y]
        
        # Фильтруем точки по порогу вероятности
        for i, keypoint in enumerate(pred_keypoints):
            presence = keypoint[0]
            if presence > threshold:
                # Денормализуем координаты обратно в пиксели
                x = int(keypoint[1] * original_image_size[1])
                y = int(keypoint[2] * original_image_size[0])
                detected_keypoints.append((i, all_keypoint_classes[i], x, y, presence))
    
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    return detected_keypoints

def visualize_keypoints(image, keypoints, heatmaps=None, model_type='regression'):
    """Визуализация обнаруженных ключевых точек и тепловых карт
    
    Args:
        image: Изображение для визуализации
        keypoints: Список обнаруженных ключевых точек
        heatmaps: Тепловые карты для визуализации (только для модели heatmap)
        model_type: Тип модели ('regression' или 'heatmap')
        
    Returns:
        Фигура matplotlib с визуализацией
    """
    # Цвета для разных групп точек
    colors = {
        'group_1': 'red',       # CP
        'group_2': 'green',     # FE2_o, FE1_o, FE2, FE1, EC1, EC2
        'group_3': 'blue'       # CT1, CT2, CD
    }
    
    if model_type == 'heatmap' and heatmaps is not None:
        # Для модели с тепловыми картами создаем визуализацию с двумя подплотами
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Первый подплот: изображение с ключевыми точками
        axes[0].imshow(image)
        axes[0].set_title("Обнаруженные ключевые точки")
        axes[0].axis('off')
        
        for idx, class_name, x, y, prob in keypoints:
            # Определяем цвет точки в зависимости от группы
            if class_name in ['CP', 'СP']:
                color = colors['group_1']
            elif class_name in ['FE2_o', 'FE1_o', 'FE2', 'FE1', 'EC1', 'EC2']:
                color = colors['group_2']
            else:
                color = colors['group_3']
            
            # Рисуем точку
            axes[0].plot(x, y, 'o', markersize=10, color=color)
            
            # Добавляем подпись с названием класса и вероятностью
            axes[0].text(x + 10, y, f"{class_name} ({prob:.2f})", fontsize=12, color=color)
        
        # Второй подплот: суммарная тепловая карта
        # Создаем суммарную тепловую карту по всем классам
        combined_heatmap = np.max(heatmaps, axis=0)
        
        # Накладываем тепловую карту на изображение
        axes[1].imshow(image)
        heatmap_resized = cv2.resize(combined_heatmap, (image.shape[1], image.shape[0]))
        axes[1].imshow(heatmap_resized, alpha=0.6, cmap='jet')
        axes[1].set_title("Тепловая карта")
        axes[1].axis('off')
        
        plt.tight_layout()
        return fig
    
    else:
        # Для модели с регрессией создаем обычную визуализацию
        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        plt.title("Обнаруженные ключевые точки")
        plt.axis('off')
        
        for idx, class_name, x, y, prob in keypoints:
            # Определяем цвет точки в зависимости от группы
            if class_name in ['CP', 'СP']:
                color = colors['group_1']
            elif class_name in ['FE2_o', 'FE1_o', 'FE2', 'FE1', 'EC1', 'EC2']:
                color = colors['group_2']
            else:
                color = colors['group_3']
            
            # Рисуем точку
            plt.plot(x, y, 'o', markersize=10, color=color)
            
            # Добавляем подпись с названием класса и вероятностью
            plt.text(x + 10, y, f"{class_name} ({prob:.2f})", fontsize=12, color=color)
        
        return plt.gcf()

def save_visualization(figure, output_path):
    """Сохранение визуализации в файл"""
    figure.savefig(output_path, bbox_inches='tight')
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description='Инференс модели обнаружения ключевых точек')
    parser.add_argument('--config', type=str, default='config.yaml', help='Путь к конфигурационному файлу')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Путь к чекпоинту модели')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению для инференса')
    parser.add_argument('--output', type=str, default='output.png', help='Путь для сохранения результата')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог вероятности для фильтрации точек')
    parser.add_argument('--model_type', type=str, default='regression', choices=['regression', 'heatmap'],
                        help='Тип модели: regression (мульти-головая регрессия) или heatmap (тепловые карты)')
    args = parser.parse_args()
    
    # Загружаем конфигурацию
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    print(f"Тип модели: {args.model_type}")
    
    # Загружаем модель
    model = load_model(args.checkpoint, model_type=args.model_type, device=device)
    print(f"Модель загружена из {args.checkpoint}")
    
    # Предобрабатываем изображение
    image_tensor, orig_image = preprocess_image(args.image)
    image_tensor = image_tensor.to(device)
    
    # Обнаруживаем ключевые точки
    with torch.no_grad():
        # Получаем предсказания модели
        outputs = model(image_tensor)
        
        # Обрабатываем выходные данные модели в зависимости от типа
        if args.model_type == 'heatmap':
            # Для модели с тепловыми картами
            heatmaps = outputs['heatmaps'][0].cpu().numpy()
            pred_keypoints = outputs['keypoints'][0].cpu().numpy()
            
            # Фильтруем точки по порогу вероятности
            keypoints = []
            for i, keypoint in enumerate(pred_keypoints):
                presence = keypoint[0]
                if presence > args.threshold:
                    # Денормализуем координаты обратно в пиксели
                    x = int(keypoint[1] * original_image_size[1])
                    y = int(keypoint[2] * original_image_size[0])
                    keypoints.append((i, all_keypoint_classes[i], x, y, presence))
        else:
            # Для модели с мульти-головой регрессией
            pred_presence = torch.sigmoid(outputs[0, :, 0]).cpu().numpy()
            pred_coords = outputs[0, :, 1:].cpu().numpy()
            heatmaps = None
            
            # Фильтруем точки по порогу вероятности
            keypoints = []
            for i, (presence, coords) in enumerate(zip(pred_presence, pred_coords)):
                if presence > args.threshold:
                    # Денормализуем координаты обратно в пиксели
                    x = int(coords[0] * original_image_size[1])
                    y = int(coords[1] * original_image_size[0])
                    keypoints.append((i, all_keypoint_classes[i], x, y, presence))
    print(f"Обнаружено {len(keypoints)} ключевых точек")
    
    # Визуализируем результаты
    figure = visualize_keypoints(orig_image, keypoints, heatmaps=heatmaps, model_type=args.model_type)
    
    # Сохраняем результат
    save_visualization(figure, args.output)
    print(f"Результат сохранен в {args.output}")
    
    # Выводим информацию о найденных точках
    print("\nОбнаруженные ключевые точки:")
    for idx, class_name, x, y, prob in keypoints:
        print(f"{class_name}: координаты ({x}, {y}), вероятность {prob:.4f}")

if __name__ == '__main__':
    main()
