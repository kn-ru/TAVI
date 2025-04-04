import torch
import torch.nn as nn
import torchvision.models as models
from dataset import group_1, group_2, group_3

class GroupKeypointModel(nn.Module):
    def __init__(self, backbone_name):
        super(GroupKeypointModel, self).__init__()
        
        # Инициализируем бэкбон в зависимости от выбранной модели
        if backbone_name.startswith('resnet'):
            # ResNet модели (resnet18, resnet34, resnet50, ...)
            self.backbone = getattr(models, backbone_name)(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.global_pool = None  # В ResNet уже есть глобальный пулинг
            
        elif backbone_name.startswith('vgg'):
            # VGG модели (vgg16, vgg19, ...)
            model = getattr(models, backbone_name)(pretrained=True)
            self.backbone = model.features  # Только сверточная часть
            self.global_pool = nn.AdaptiveAvgPool2d(1)  # Добавляем глобальный пулинг
            # Для VGG16 и VGG19 размер вектора признаков после пулинга - 512
            num_features = 512
            
        elif backbone_name.startswith('densenet'):
            # DenseNet модели (densenet121, densenet169, ...)
            self.backbone = getattr(models, backbone_name)(pretrained=True)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.global_pool = None
            
        elif backbone_name.startswith('efficientnet'):
            # EfficientNet модели
            self.backbone = getattr(models, backbone_name)(pretrained=True)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.global_pool = None
            
        elif backbone_name.startswith('vit'):
            # Vision Transformer модели (vit_b_16, vit_b_32, vit_l_16, ...)
            self.backbone = getattr(models, backbone_name)(pretrained=True)
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Identity()
            self.global_pool = None
            
            # Добавляем ресайз для ViT, так как он ожидает изображения размером 224x224
            self.resize = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
            
        else:
            # По умолчанию используем ResNet34
            self.backbone = models.resnet34(pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.global_pool = None
            
        # Запоминаем тип бэкбона для использования в forward
        self.backbone_name = backbone_name
        
        # Общие признаки для всех групп
        self.shared_features = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.LayerNorm(512),  # Заменяем BatchNorm1d на LayerNorm, который работает с любым размером батча
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Отдельная голова для классификации групп
        self.group_classifier = nn.Linear(512, 3)  # 3 группы точек
        
        # Отдельные головы для каждой группы точек
        self.group1_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * len(group_1))  # (p, x, y) для каждой точки в группе 1
        )
        
        self.group2_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * len(group_2))  # (p, x, y) для каждой точки в группе 2
        )
        
        self.group3_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * len(group_3))  # (p, x, y) для каждой точки в группе 3
        )
        
        # Сохраняем размеры групп
        self.group1_size = len(group_1)
        self.group2_size = len(group_2)
        self.group3_size = len(group_3)
        self.num_keypoints = self.group1_size + self.group2_size + self.group3_size
    

    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Извлекаем признаки из бэкбона в зависимости от типа модели
        if self.backbone_name.startswith('vgg'):
            # Для VGG нужно сначала получить feature maps, затем применить глобальный пулинг
            features = self.backbone(x)
            pooled = self.global_pool(features)
            backbone_features = pooled.view(batch_size, -1)  # Преобразуем в вектор
        elif self.backbone_name.startswith('vit'):
            # Для Vision Transformer сначала изменяем размер изображения до 224x224
            # затем пропускаем через модель
            x_resized = self.resize(x)
            backbone_features = self.backbone(x_resized)
        else:
            # Для остальных моделей (ResNet, DenseNet, EfficientNet) просто получаем вектор признаков
            backbone_features = self.backbone(x)
        
        # Получаем общие признаки
        shared = self.shared_features(backbone_features)
        
        # Предсказываем вероятности групп
        group_probs = self.group_classifier(shared)
        
        # Предсказываем точки для каждой группы
        group1_out = self.group1_head(shared).view(-1, self.group1_size, 3)
        group2_out = self.group2_head(shared).view(-1, self.group2_size, 3)
        group3_out = self.group3_head(shared).view(-1, self.group3_size, 3)
        
        # Объединяем выходы всех групп
        keypoints = torch.cat([group1_out, group2_out, group3_out], dim=1)
        
        return {
            'keypoints': keypoints,      # [batch, num_keypoints, 3]
            'group_probs': group_probs,  # [batch, 3]
            'group1': group1_out,        # [batch, group1_size, 3]
            'group2': group2_out,        # [batch, group2_size, 3]
            'group3': group3_out         # [batch, group3_size, 3]
        }


# Для обратной совместимости с существующим кодом
class MultiHeadKeypointModel(GroupKeypointModel):
    def __init__(self, num_keypoints, backbone_name='resnet18'):
        super(MultiHeadKeypointModel, self).__init__(backbone_name=backbone_name)
        
    def forward(self, x):
        output = super(MultiHeadKeypointModel, self).forward(x)
        return output['keypoints']  # Возвращаем только предсказания точек для совместимости
