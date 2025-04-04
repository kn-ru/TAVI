import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataset import group_1, group_2, group_3, all_keypoint_classes

class KeypointLoss(nn.Module):
    """
    Комбинированная функция потерь для задачи обнаружения ключевых точек.
    Включает:
    1. BCE Loss для предсказания наличия точки
    2. Усиленный L1/L2 Loss для предсказания координат
    3. Focal Loss для улучшения работы с несбалансированными данными
    """
    def __init__(self, lambda_coord=5.0, lambda_group=0.5, focal_alpha=0.25, focal_gamma=2.0, use_wing_loss=True):
        super(KeypointLoss, self).__init__()
        self.lambda_coord = lambda_coord  # Вес для координатной ошибки
        self.lambda_group = lambda_group  # Вес для ошибки классификации групп
        self.focal_alpha = focal_alpha    # Параметр alpha для Focal Loss
        self.focal_gamma = focal_gamma    # Параметр gamma для Focal Loss
        self.use_wing_loss = use_wing_loss  # Использовать Wing Loss вместо L1
        self.wing_w = 10.0                # Параметр w для Wing Loss
        self.wing_epsilon = 2.0           # Параметр epsilon для Wing Loss
        
    def wing_loss(self, pred, target, w=10.0, epsilon=2.0):
        """
        Wing Loss для более точной регрессии координат.
        Ссылка: https://arxiv.org/abs/1711.06753
        """
        c = w * (1.0 - np.log(1.0 + w/epsilon))
        abs_diff = torch.abs(pred - target)
        
        loss = torch.where(
            abs_diff < w,
            w * torch.log(1.0 + abs_diff / epsilon),
            abs_diff - c
        )
        
        return loss
    
    def focal_loss(self, pred, target):
        """
        Focal Loss для улучшения классификации наличия точек.
        Ссылка: https://arxiv.org/abs/1708.02002
        """
        # Применяем сигмоиду к предсказаниям
        pred_prob = torch.sigmoid(pred)
        
        # Вычисляем binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        
        # Вычисляем focal weights
        p_t = pred_prob * target + (1 - pred_prob) * (1 - target)
        focal_weight = (1 - p_t) ** self.focal_gamma
        
        # Применяем alpha
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)
        
        # Итоговый focal loss
        focal = alpha_t * focal_weight * bce
        
        return focal
    
    def forward(self, outputs, targets, group_targets=None):
        """
        Вычисление комбинированной функции потерь.
        
        Args:
            outputs: Словарь с выходами модели или тензор [batch, num_keypoints, 3]
            targets: Тензор с ground truth [batch, num_keypoints, 3]
            group_targets: Тензор с метками групп [batch, 3] (опционально)
            
        Returns:
            Словарь с отдельными компонентами потерь и общей потерей
        """
        # Обрабатываем разные форматы выхода модели
        if isinstance(outputs, dict):
            pred_keypoints = outputs['keypoints']
            pred_groups = outputs.get('group_probs', None)
        else:
            pred_keypoints = outputs
            pred_groups = None
        
        # Разделяем предсказания и ground truth
        pred_presence = pred_keypoints[:, :, 0]  # [batch, num_keypoints]
        pred_coords = pred_keypoints[:, :, 1:]   # [batch, num_keypoints, 2]
        
        gt_presence = targets[:, :, 0]  # [batch, num_keypoints]
        gt_coords = targets[:, :, 1:]   # [batch, num_keypoints, 2]
        
        # 1. Потеря для предсказания наличия точки (Focal Loss)
        presence_loss = self.focal_loss(pred_presence, gt_presence).mean()
        
        # 2. Потеря для предсказания координат
        # Создаем маску для точек, которые присутствуют
        presence_mask = gt_presence.unsqueeze(2).expand_as(pred_coords)  # [batch, num_keypoints, 2]
        
        # Вычисляем ошибку координат только для присутствующих точек
        if self.use_wing_loss:
            coord_error = self.wing_loss(pred_coords, gt_coords, self.wing_w, self.wing_epsilon)
        else:
            coord_error = F.smooth_l1_loss(pred_coords, gt_coords, reduction='none')
        
        # Применяем маску и нормализуем
        masked_coord_error = coord_error * presence_mask
        num_present = presence_mask.sum()
        
        if num_present > 0:
            coord_loss = masked_coord_error.sum() / num_present
        else:
            coord_loss = torch.tensor(0.0, device=pred_coords.device)
        
        # 3. Потеря для классификации групп
        # Используем вместо меток групп информацию о наличии точек в группах
        group_loss = torch.tensor(0.0, device=pred_coords.device)
        
        if pred_groups is not None:
            # Создаем псевдо-метки для групп на основе наличия точек
            batch_size = gt_presence.size(0)
            
            # Группа 1: точки CP (1 точка)
            group1_indices = [all_keypoint_classes.index(cls) for cls in group_1]
            group1_presence = gt_presence[:, group1_indices].sum(dim=1) > 0
            
            # Группа 2: точки FE2_o, FE1_o, FE2, FE1, EC1, EC2 (6 точек)
            group2_indices = [all_keypoint_classes.index(cls) for cls in group_2]
            group2_presence = gt_presence[:, group2_indices].sum(dim=1) > 0
            
            # Группа 3: точки CT1, CT2, CD (3 точки)
            group3_indices = [all_keypoint_classes.index(cls) for cls in group_3]
            group3_presence = gt_presence[:, group3_indices].sum(dim=1) > 0
            
            # Создаем веса для каждой группы
            group_weights = torch.stack([group1_presence.float(), 
                                      group2_presence.float(), 
                                      group3_presence.float()], dim=1)
            
            # Используем многометочную классификацию вместо однометочной
            group_loss = F.binary_cross_entropy_with_logits(pred_groups, group_weights)
        
        # Общая потеря
        total_loss = presence_loss + self.lambda_coord * coord_loss + self.lambda_group * group_loss
        
        # Возвращаем словарь с компонентами потерь для мониторинга
        return {
            'total': total_loss,
            'presence': presence_loss,
            'coord': coord_loss,
            'group': group_loss
        }
