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
from model import MultiHeadKeypointModel
from dataset import all_keypoint_classes, image_size, original_image_size

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
    
    # Фильтруем точки по порогу вероятности
    detected_keypoints = []
    for i, (presence, coords) in enumerate(zip(pred_presence, pred_coords)):
        if presence > threshold:
            # Денормализуем координаты обратно в пиксели
            x = int(coords[0] * original_image_size[1])
            y = int(coords[1] * original_image_size[0])
            detected_keypoints.append((i, all_keypoint_classes[i], x, y, presence))
    
    return detected_keypoints

def visualize_keypoints(image, keypoints):
    """Визуализация обнаруженных ключевых точек"""
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    
    # Цвета для разных групп точек
    colors = {
        'group_1': 'red',       # CP
        'group_2': 'green',     # FE2_o, FE1_o, FE2, FE1, EC1, EC2
        'group_3': 'blue'       # CT1, CT2, CD
    }
    
    for idx, class_name, x, y, prob in keypoints:
        # Определяем цвет точки в зависимости от группы
        if class_name in ['CP']:
            color = colors['group_1']
        elif class_name in ['FE2_o', 'FE1_o', 'FE2', 'FE1', 'EC1', 'EC2']:
            color = colors['group_2']
        else:
            color = colors['group_3']
        
        # Рисуем точку
        plt.plot(x, y, 'o', markersize=10, color=color)
        
        # Добавляем подпись с названием класса и вероятностью
        plt.text(x + 10, y, f"{class_name} ({prob:.2f})", fontsize=12, color=color)
    
    plt.title("Обнаруженные ключевые точки")
    plt.axis('off')
    
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
    
    # Предобрабатываем изображение
    image_tensor, orig_image = preprocess_image(args.image)
    image_tensor = image_tensor.to(device)
    
    # Обнаруживаем ключевые точки
    keypoints = detect_keypoints(model, image_tensor, args.threshold)
    print(f"Обнаружено {len(keypoints)} ключевых точек")
    
    # Визуализируем результаты
    figure = visualize_keypoints(orig_image, keypoints)
    
    # Сохраняем результат
    save_visualization(figure, args.output)
    print(f"Результат сохранен в {args.output}")
    
    # Выводим информацию о найденных точках
    print("\nОбнаруженные ключевые точки:")
    for idx, class_name, x, y, prob in keypoints:
        print(f"{class_name}: координаты ({x}, {y}), вероятность {prob:.4f}")

if __name__ == '__main__':
    main()
