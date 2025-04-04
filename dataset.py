import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import json
import numpy as np
import cv2
from PIL import Image

# Пути к данным
original_image_size = (1024, 1024)  # Исходное разрешение изображений
image_size = (512, 512)  # Новое разрешение изображений

# Классы ключевых точек
group_1 = ["CP"]  # Проксимальный конец катетера (1 точка)
group_2 = ["FE2_o", "FE1_o", "FE2", "FE1", "EC1", "EC2"]  # Синотубулярное соединение (много точек)
group_3 = ["CT1", "CT2", "CD"]  # Дистальный конец катетера и конец катетера (от 0 до 3 точек)
ignore_classes = ["CM"]  # Мусорные точки (не учитывать)

# Все классы точек
all_keypoint_classes = group_1 + group_2 + group_3
num_keypoints = len(all_keypoint_classes)

# Параметры аугментаций
scale_factor = 0.3    # Диапазон случайного масштабирования (увеличен)
rotation_factor = 40   # Максимальный угол поворота (увеличен)
flip_prob = 0.5       # Вероятность горизонтального флипа
blur_prob = 0.3       # Вероятность применения размытия
blur_radius = (1, 3)  # Диапазон радиуса размытия
noise_prob = 0.2      # Вероятность добавления шума
noise_scale = 0.05    # Максимальная интенсивность шума
brightness_range = (0.8, 1.2)  # Диапазон изменения яркости
contrast_range = (0.8, 1.2)    # Диапазон изменения контрастности


def get_affine_transform(center, scale, rot, output_size):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale * 200.0

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.])
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def flip_back(output, pairs):
    output = output[:, :, ::-1]

    for pair in pairs:
        tmp = output[:, pair[0], :].copy()
        output[:, pair[0], :] = output[:, pair[1], :]
        output[:, pair[1], :] = tmp

    return output


def flip_joints(joints, width, pairs):
    joints[:, 0] = width - joints[:, 0] - 1

    for pair in pairs:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()

    return joints


def apply_blur(image, prob=blur_prob, radius_range=blur_radius):
    """Применяет размытие к изображению с заданной вероятностью"""
    if np.random.random() < prob:
        # Выбираем случайный радиус размытия из диапазона
        radius = np.random.randint(radius_range[0], radius_range[1] + 1)
        # Применяем размытие по Гауссу
        return cv2.GaussianBlur(image, (radius * 2 + 1, radius * 2 + 1), 0)
    return image


def apply_noise(image, prob=noise_prob, scale=noise_scale):
    """Добавляет гауссовский шум к изображению с заданной вероятностью"""
    if np.random.random() < prob:
        # Генерируем гауссовский шум
        noise = np.random.normal(0, scale, image.shape).astype(np.float32)
        # Добавляем шум к изображению
        noisy_image = image.astype(np.float32) + noise
        # Ограничиваем значения в диапазоне [0, 255]
        return np.clip(noisy_image, 0, 255).astype(image.dtype)
    return image


def adjust_brightness(image, brightness_range=brightness_range):
    """Изменяет яркость изображения"""
    # Случайный коэффициент яркости из диапазона
    brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
    # Применяем изменение яркости
    image_float = image.astype(np.float32)
    image_float = image_float * brightness_factor
    # Ограничиваем значения в диапазоне [0, 255]
    return np.clip(image_float, 0, 255).astype(image.dtype)


def adjust_contrast(image, contrast_range=contrast_range):
    """Изменяет контрастность изображения"""
    # Случайный коэффициент контрастности из диапазона
    contrast_factor = np.random.uniform(contrast_range[0], contrast_range[1])
    # Вычисляем среднюю яркость изображения
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    # Применяем изменение контрастности
    image_float = image.astype(np.float32)
    image_float = mean + contrast_factor * (image_float - mean)
    # Ограничиваем значения в диапазоне [0, 255]
    return np.clip(image_float, 0, 255).astype(image.dtype)


# Функция для загрузки JSON и генерации координат точек
def extract_points_from_json(json_path):
    """Extract keypoint coordinates from JSON annotation file"""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Инициализируем массив для всех ключевых точек
    # Формат: [presence, x, y] для каждой точки
    keypoints = np.zeros((num_keypoints, 3), dtype=np.float32)
    
    # Заполняем массив данными из JSON
    for obj in data["objects"]:
        class_name = obj["classTitle"]
        if class_name in ignore_classes:
            continue  # Пропускаем мусорные точки
        
        if class_name in all_keypoint_classes:
            idx = all_keypoint_classes.index(class_name)
            x, y = obj["points"]["exterior"][0]
            
            # Нормализуем координаты к [0, 1]
            x_norm = x / original_image_size[1]
            y_norm = y / original_image_size[0]
            
            # Устанавливаем наличие точки и её координаты
            keypoints[idx, 0] = 1.0  # presence flag
            keypoints[idx, 1] = x_norm  # normalized x
            keypoints[idx, 2] = y_norm  # normalized y
    
    return keypoints


# Класс датасета
class TAVIDataset(Dataset):
    def __init__(self, root_dataset, mode='train', transform=None):
        self.image_paths = []
        self.json_paths = []
        self.transform = transform
        self.mode = mode
        
        # Определяем, какие папки использовать для обучения и валидации
        all_folders = sorted(glob.glob(os.path.join(root_dataset, "*")))
        if mode == 'train':
            folders = all_folders[:int(0.8 * len(all_folders))]  # 80% для обучения
        else:
            folders = all_folders[int(0.8 * len(all_folders)):]  # 20% для валидации

        for case_folder in folders:
            if not os.path.isdir(case_folder):
                continue
                
            ann_folder = os.path.join(case_folder, "ann")
            img_folder = os.path.join(case_folder, "img")
            
            if not os.path.exists(ann_folder) or not os.path.exists(img_folder):
                continue

            json_files = sorted(glob.glob(os.path.join(ann_folder, "*.png.json")))
            
            for json_file in json_files:
                img_name = os.path.basename(json_file).replace(".json", "")
                img_path = os.path.join(img_folder, img_name)
                
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.json_paths.append(json_file)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        json_path = self.json_paths[idx]
        
        # Загружаем изображение
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Получаем координаты ключевых точек
        keypoints = extract_points_from_json(json_path)
        
        # Применяем аугментации, если нужно
        if self.transform and self.mode == 'train':
            # Применяем изменение яркости и контрастности
            image = adjust_brightness(image)
            image = adjust_contrast(image)
            
            # Применяем размытие и шум
            image = apply_blur(image)
            image = apply_noise(image)
            
            # Центр изображения
            c = np.array([image.shape[1] / 2.0, image.shape[0] / 2.0])
            s = max(image.shape[0], image.shape[1]) / 200.0
            
            # Случайное масштабирование и поворот для аугментации
            sf = scale_factor
            rf = rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            # Увеличиваем вероятность поворота до 80%
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if np.random.random() < 0.8 else 0
            
            # Случайное отражение
            flip = np.random.random() < flip_prob
            
            # Применяем отражение к изображению
            if flip:
                image = image[:, ::-1, :]
                
                # Применяем отражение к координатам точек
                for i in range(num_keypoints):
                    if keypoints[i, 0] > 0:  # Если точка присутствует
                        keypoints[i, 1] = 1.0 - keypoints[i, 1]  # Отражаем x-координату
            
            # Получаем матрицу преобразования
            trans = get_affine_transform(c, s, r, image_size)
            
            # Применяем аффинное преобразование к изображению
            image = cv2.warpAffine(
                image, trans, (int(image_size[1]), int(image_size[0])),
                flags=cv2.INTER_LINEAR
            )
            
            # Применяем аффинное преобразование к координатам точек
            for i in range(num_keypoints):
                if keypoints[i, 0] > 0:  # Если точка присутствует
                    # Денормализуем координаты обратно в пиксели
                    x = keypoints[i, 1] * original_image_size[1]
                    y = keypoints[i, 2] * original_image_size[0]
                    
                    # Применяем аффинное преобразование
                    x, y = affine_transform([x, y], trans)
                    
                    # Нормализуем обратно в диапазон [0, 1]
                    keypoints[i, 1] = x / image_size[1]
                    keypoints[i, 2] = y / image_size[0]
                    
                    # Проверяем, что точка все еще в пределах изображения
                    if x < 0 or y < 0 or x >= image_size[1] or y >= image_size[0]:
                        keypoints[i, 0] = 0  # Точка вышла за пределы изображения
        else:
            # Просто изменяем размер изображения без аугментаций
            image = cv2.resize(image, (image_size[1], image_size[0]))
        
        # Нормализуем изображение
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Преобразуем в тензор с явным указанием типа float32
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        keypoints = torch.from_numpy(keypoints).float()
        
        return image, keypoints
