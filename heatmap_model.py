import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import all_keypoint_classes

class HeatmapKeypointModel(nn.Module):
    """
    Модель для обнаружения ключевых точек на основе тепловых карт.
    Для каждой ключевой точки генерируется отдельный канал тепловой карты.
    """
    def __init__(self, backbone_type='hrnet', heatmap_size=(64, 64), sigma=2.0):
        super(HeatmapKeypointModel, self).__init__()
        self.num_keypoints = len(all_keypoint_classes)
        self.heatmap_size = heatmap_size
        self.sigma = sigma  # Стандартное отклонение для гауссова распределения
        
        # Выбираем базовую архитектуру
        if backbone_type == 'hrnet':
            from hrnet_model import HRNetKeypointModel
            self.backbone = HRNetKeypointModel(width=32)
            
            # Заменяем последний слой для генерации тепловых карт
            # Вместо выходного слоя с 3 каналами (наличие, x, y) для каждой точки
            # делаем слой с num_keypoints каналами (по одному на каждую точку)
            # В HRNet после объединения всех разрешений получается 480 каналов
            # Получается из суммы каналов всех ветвей: 32 + 64 + 128 + 256 = 480
            self.heatmap_head = nn.Conv2d(
                in_channels=480,
                out_channels=self.num_keypoints,
                kernel_size=1,
                stride=1,
                padding=0
            )
            
            # Удаляем финальный слой из HRNet, так как мы заменяем его на наш
            delattr(self.backbone, 'final_layer')
        else:
            raise ValueError(f"Неподдерживаемый тип backbone: {backbone_type}")
    
    def forward(self, x):
        """
        Прямой проход модели.
        
        Args:
            x: Входное изображение [batch_size, 3, H, W]
            
        Returns:
            dict: Словарь с выходными данными модели
                - heatmaps: Предсказанные тепловые карты [batch_size, num_keypoints, heatmap_H, heatmap_W]
                - keypoints: Предсказанные координаты ключевых точек [batch_size, num_keypoints, 3]
                  где 3 канала - это [presence, x, y]
        """
        batch_size = x.size(0)
        
        # Выполняем часть backbone до final_layer
        # Мы удалили final_layer из backbone, поэтому нам нужно повторить логику из HRNetKeypointModel
        # До места, где используется final_layer
        
        # Начальные слои
        x = self.backbone.conv1(x)
        x = self.backbone.conv2(x)
        
        # Stage 1
        x = self.backbone.stage1_branch(x)
        x_list = [x]
        
        # Transition 1
        x_list_stage2 = []
        for i in range(len(self.backbone.transition1)):
            if self.backbone.transition1[i] is not None:
                x_list_stage2.append(self.backbone.transition1[i](x_list[-1]))
            else:
                x_list_stage2.append(x_list[i])
        
        # Stage 2
        x_list = self.backbone.stage2(x_list_stage2)
        
        # Transition 2
        x_list_stage3 = []
        for i in range(len(self.backbone.transition2)):
            if self.backbone.transition2[i] is not None:
                x_list_stage3.append(self.backbone.transition2[i](x_list[-1]))
            else:
                x_list_stage3.append(x_list[i])
        
        # Stage 3
        x_list = self.backbone.stage3(x_list_stage3)
        
        # Transition 3
        x_list_stage4 = []
        for i in range(len(self.backbone.transition3)):
            if self.backbone.transition3[i] is not None:
                x_list_stage4.append(self.backbone.transition3[i](x_list[-1]))
            else:
                x_list_stage4.append(x_list[i])
        
        # Stage 4
        x_list = self.backbone.stage4(x_list_stage4)
        
        # Объединяем все разрешения
        x0_h, x0_w = x_list[0].size(2), x_list[0].size(3)
        x_all = [x_list[0]]
        for i in range(1, len(x_list)):
            x_all.append(F.interpolate(x_list[i], size=(x0_h, x0_w), mode='bilinear', align_corners=False))
        x = torch.cat(x_all, dim=1)
        
        # Вместо final_layer используем наш heatmap_head
        # Для тепловых карт нам не нужен global_pool
        heatmaps = self.heatmap_head(x)
        
        # Применяем сигмоиду для нормализации значений в диапазоне [0, 1]
        heatmaps = torch.sigmoid(heatmaps)
        
        # Извлекаем координаты ключевых точек из тепловых карт
        keypoints = self.extract_keypoints_from_heatmaps(heatmaps)
        
        return {
            'heatmaps': heatmaps,
            'keypoints': keypoints
        }
    
    def extract_keypoints_from_heatmaps(self, heatmaps):
        """
        Извлекает координаты ключевых точек из тепловых карт.
        
        Args:
            heatmaps: Тепловые карты [batch_size, num_keypoints, H, W]
            
        Returns:
            torch.Tensor: Тензор с координатами ключевых точек [batch_size, num_keypoints, 3]
                где 3 канала - это [presence, x, y]
        """
        batch_size = heatmaps.size(0)
        
        # Создаем тензор для хранения координат
        keypoints = torch.zeros(batch_size, self.num_keypoints, 3, device=heatmaps.device)
        
        # Находим максимальные значения и их индексы для каждой тепловой карты
        max_values, _ = torch.max(heatmaps.view(batch_size, self.num_keypoints, -1), dim=2)
        
        # Находим индексы максимальных значений в 2D пространстве
        for b in range(batch_size):
            for k in range(self.num_keypoints):
                heatmap = heatmaps[b, k]
                
                # Находим максимальное значение и его индекс
                max_value = torch.max(heatmap)
                
                # Устанавливаем порог для определения наличия точки
                # Если максимальное значение меньше порога, считаем, что точки нет
                presence = 1.0 if max_value > 0.3 else 0.0
                
                if presence > 0:
                    # Находим координаты максимума
                    max_index = torch.argmax(heatmap.view(-1))
                    y, x = max_index // heatmap.size(1), max_index % heatmap.size(1)
                    
                    # Уточняем координаты с помощью метода центра масс вокруг пика
                    x_refined, y_refined = self.refine_coordinates(heatmap, x, y)
                    
                    # Нормализуем координаты в диапазоне [0, 1]
                    x_norm = x_refined / heatmap.size(1)
                    y_norm = y_refined / heatmap.size(0)
                    
                    # Сохраняем результаты
                    keypoints[b, k, 0] = presence
                    keypoints[b, k, 1] = x_norm
                    keypoints[b, k, 2] = y_norm
        
        return keypoints
    
    def refine_coordinates(self, heatmap, x, y, window_size=3):
        """
        Уточняет координаты с помощью метода центра масс вокруг пика.
        
        Args:
            heatmap: Тепловая карта [H, W]
            x, y: Координаты пика
            window_size: Размер окна для вычисления центра масс
            
        Returns:
            tuple: Уточненные координаты (x, y)
        """
        h, w = heatmap.size()
        
        # Определяем границы окна
        left = max(0, x - window_size // 2)
        right = min(w - 1, x + window_size // 2)
        top = max(0, y - window_size // 2)
        bottom = min(h - 1, y + window_size // 2)
        
        # Вырезаем окно вокруг пика
        window = heatmap[top:bottom+1, left:right+1]
        
        # Создаем сетку координат
        y_grid, x_grid = torch.meshgrid(
            torch.arange(top, bottom+1, device=heatmap.device),
            torch.arange(left, right+1, device=heatmap.device)
        )
        
        # Вычисляем центр масс
        sum_values = torch.sum(window)
        
        if sum_values > 0:
            x_refined = torch.sum(x_grid * window) / sum_values
            y_refined = torch.sum(y_grid * window) / sum_values
        else:
            x_refined = torch.tensor(x, dtype=torch.float, device=heatmap.device)
            y_refined = torch.tensor(y, dtype=torch.float, device=heatmap.device)
        
        return x_refined, y_refined
    
    def generate_target_heatmaps(self, keypoints, heatmap_size=None):
        """
        Генерирует целевые тепловые карты из координат ключевых точек.
        
        Args:
            keypoints: Тензор с координатами ключевых точек [batch_size, num_keypoints, 3]
                где 3 канала - это [presence, x, y]
            heatmap_size: Размер выходной тепловой карты (H, W)
            
        Returns:
            torch.Tensor: Тензор с тепловыми картами [batch_size, num_keypoints, H, W]
        """
        if heatmap_size is None:
            heatmap_size = self.heatmap_size
            
        batch_size = keypoints.size(0)
        
        # Создаем тензор для хранения тепловых карт
        target_heatmaps = torch.zeros(
            batch_size, self.num_keypoints, heatmap_size[0], heatmap_size[1],
            device=keypoints.device
        )
        
        # Генерируем тепловые карты для каждой точки
        for b in range(batch_size):
            for k in range(self.num_keypoints):
                # Проверяем наличие точки
                if keypoints[b, k, 0] > 0:
                    # Получаем координаты точки (нормализованные в [0, 1])
                    x_norm, y_norm = keypoints[b, k, 1], keypoints[b, k, 2]
                    
                    # Преобразуем в координаты на тепловой карте
                    x = x_norm * heatmap_size[1]
                    y = y_norm * heatmap_size[0]
                    
                    # Генерируем гауссово распределение вокруг точки
                    target_heatmaps[b, k] = self.generate_gaussian_heatmap(
                        heatmap_size, (x, y), self.sigma
                    )
        
        return target_heatmaps
    
    def generate_gaussian_heatmap(self, size, point, sigma):
        """
        Генерирует гауссово распределение вокруг точки.
        
        Args:
            size: Размер тепловой карты (H, W)
            point: Координаты центра (x, y)
            sigma: Стандартное отклонение
            
        Returns:
            torch.Tensor: Тепловая карта с гауссовым распределением [H, W]
        """
        x, y = point
        height, width = size
        
        # Создаем сетку координат
        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            torch.arange(width, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        )
        
        # Вычисляем расстояние от каждой точки до центра
        d2 = (x_grid - x) ** 2 + (y_grid - y) ** 2
        
        # Генерируем гауссово распределение
        exponent = d2 / (2 * sigma ** 2)
        heatmap = torch.exp(-exponent)
        
        return heatmap
