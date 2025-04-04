import os
import argparse
import cv2
import numpy as np
import json
# Устанавливаем Agg backend для matplotlib, чтобы избежать проблем с Qt
import matplotlib
matplotlib.use('Agg')  # Важно установить до импорта pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def parse_args():
    parser = argparse.ArgumentParser(description='Визуализация GT разметки на изображении')
    parser.add_argument('--image_path', type=str, required=True, 
                        help='Путь к изображению для визуализации')
    parser.add_argument('--output_path', type=str, default='gt_visualization.png',
                        help='Путь для сохранения результата визуализации')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Проверяем, существует ли изображение
    if not os.path.exists(args.image_path):
        print(f"Ошибка: Изображение {args.image_path} не найдено")
        return
    
    # Получаем путь к JSON-файлу с аннотациями
    json_path = args.image_path.replace('/img/', '/ann/') + '.json'
    if not os.path.exists(json_path):
        print(f"Ошибка: JSON-файл с аннотациями {json_path} не найден")
        return
    
    # Загружаем изображение
    img = mpimg.imread(args.image_path)
    
    # Загружаем разметку напрямую из JSON
    with open(json_path, 'r') as f:
        annotations = json.load(f)
    
    # Создаем фигуру для визуализации
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='gray')
    plt.title(f'Ground Truth Keypoints: {os.path.basename(args.image_path)}')
    
    # Цвета для каждой метки
    colors = {
        # Добавляем оба варианта - с русской и английской буквой C/С
        'СT1': 'purple', 'CT1': 'purple', # русская С и английская C
        'CT2': 'orange', 'СT2': 'orange',
        'CD': 'pink', 'СD': 'pink',
        'CP': 'brown', 'СP': 'brown',
        'FE2_o': 'yellow', 'FE1_o': 'green', 
        'FE2': 'blue', 'FE1': 'red', 
        'EC1': 'cyan', 'EC2': 'black', 
        'CM': 'black', 'СM': 'black'  # также обрабатываем оба варианта CM
    }
    
    # Нанесение точек на изображение
    for obj in annotations['objects']:
        class_title = obj['classTitle']
        points = obj['points']['exterior']
        x, y = points[0]
        
        # Обработка случая, если класс не найден в словаре цветов
        if class_title not in colors:
            print(f"\u0412нимание: Неизвестный класс '{class_title}'. Используем серый цвет.")
            color = 'gray'
        else:
            color = colors[class_title]
            
        plt.scatter(x, y, c=color, label=class_title, s=100)
        plt.text(x+5, y+5, class_title, color=color, fontsize=10)
    
    plt.axis('off')
    
    # Добавляем легенду без дубликатов
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    
    # Сохраняем результат
    plt.tight_layout()
    plt.savefig(args.output_path)
    print(f"Результат сохранен в {args.output_path}")
    
    # Не показываем изображение, только сохраняем
    # plt.show()

if __name__ == "__main__":
    main()
