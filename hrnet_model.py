import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import group_1, group_2, group_3

class ConvBlock(nn.Module):
    """Базовый сверточный блок с batch normalization и ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    """Residual блок для HRNet"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class HighResolutionModule(nn.Module):
    """Основной модуль HRNet, который обрабатывает несколько разрешений параллельно"""
    def __init__(self, num_branches, blocks, num_blocks, num_channels, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output
        
        self.branches = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers(num_branches, num_channels, multi_scale_output)
        self.relu = nn.ReLU(inplace=True)
    
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        
        for i in range(num_branches):
            layers = []
            for j in range(num_blocks[i]):
                layers.append(block(num_channels[i], num_channels[i]))
            
            branches.append(nn.Sequential(*layers))
        
        return nn.ModuleList(branches)
    
    def _make_fuse_layers(self, num_branches, num_channels, multi_scale_output=True):
        if num_branches == 1:
            return None
        
        num_branches_out = num_branches if multi_scale_output else 1
        fuse_layers = []
        
        for i in range(num_branches_out):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    # Upsample: j -> i (j имеет более низкое разрешение, чем i)
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels[j], num_channels[i], kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(num_channels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=False)
                        )
                    )
                elif j == i:
                    # Тот же уровень разрешения
                    fuse_layer.append(None)
                else:
                    # Downsample: j -> i (j имеет более высокое разрешение, чем i)
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(num_channels[j], num_channels[i], kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(num_channels[i])
                                )
                            )
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(num_channels[j], num_channels[j], kernel_size=3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(num_channels[j]),
                                    nn.ReLU(inplace=True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            
            fuse_layers.append(nn.ModuleList(fuse_layer))
        
        return nn.ModuleList(fuse_layers)
    
    def forward(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        
        if self.fuse_layers is not None:
            x_fuse = []
            for i in range(len(self.fuse_layers)):
                y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
                for j in range(1, self.num_branches):
                    if i == j:
                        y = y + x[j]
                    else:
                        y = y + self.fuse_layers[i][j](x[j])
                x_fuse.append(self.relu(y))
            x = x_fuse
        
        return x

class HRNetKeypointModel(nn.Module):
    """HRNet модель для детекции ключевых точек"""
    def __init__(self, width=32):
        super(HRNetKeypointModel, self).__init__()
        
        # Начальные слои
        self.conv1 = ConvBlock(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBlock(64, 64, kernel_size=3, stride=2, padding=1)
        
        # Stage 1
        self.stage1_branch = self._make_layer(BasicBlock, 64, width, 4)
        
        # Transition 1: от 1 ветви к 2 ветвям
        self.transition1 = self._make_transition_layer([width], [width, width*2])
        
        # Stage 2
        num_channels = [width, width*2]
        num_branches = 2
        num_blocks = [4, 4]
        self.stage2 = HighResolutionModule(num_branches, BasicBlock, num_blocks, num_channels)
        
        # Transition 2: от 2 ветвей к 3 ветвям
        self.transition2 = self._make_transition_layer(num_channels, [width, width*2, width*4])
        
        # Stage 3
        num_channels = [width, width*2, width*4]
        num_branches = 3
        num_blocks = [4, 4, 4]
        self.stage3 = HighResolutionModule(num_branches, BasicBlock, num_blocks, num_channels)
        
        # Transition 3: от 3 ветвей к 4 ветвям
        self.transition3 = self._make_transition_layer(num_channels, [width, width*2, width*4, width*8])
        
        # Stage 4
        num_channels = [width, width*2, width*4, width*8]
        num_branches = 4
        num_blocks = [4, 4, 4, 4]
        self.stage4 = HighResolutionModule(num_branches, BasicBlock, num_blocks, num_channels, multi_scale_output=True)
        
        # Финальный слой для объединения всех разрешений
        self.final_layer = nn.Sequential(
            nn.Conv2d(sum(num_channels), 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Глобальный пулинг
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Общие признаки для всех групп
        self.shared_features = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512),
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
    
    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        transition_layers = []
        for i in range(len(num_channels_cur_layer)):
            if i < len(num_channels_pre_layer):
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], kernel_size=3, stride=1, padding=1, bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i - len(num_channels_pre_layer) + 1):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] if j == i - len(num_channels_pre_layer) else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv_downsamples))
        
        return nn.ModuleList(transition_layers)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Начальные слои
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Stage 1
        x = self.stage1_branch(x)
        
        # Transition 1
        x_list = []
        for i in range(len(self.transition1)):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        
        # Stage 2
        x_list = self.stage2(x_list)
        
        # Transition 2
        x_list_stage3 = []
        for i in range(len(self.transition2)):
            if self.transition2[i] is not None:
                x_list_stage3.append(self.transition2[i](x_list[-1]))
            else:
                x_list_stage3.append(x_list[i])
        
        # Stage 3
        x_list = self.stage3(x_list_stage3)
        
        # Transition 3
        x_list_stage4 = []
        for i in range(len(self.transition3)):
            if self.transition3[i] is not None:
                x_list_stage4.append(self.transition3[i](x_list[-1]))
            else:
                x_list_stage4.append(x_list[i])
        
        # Stage 4
        x_list = self.stage4(x_list_stage4)
        
        # Объединяем все разрешения
        x0_h, x0_w = x_list[0].size(2), x_list[0].size(3)
        x_all = [x_list[0]]
        for i in range(1, len(x_list)):
            x_all.append(F.interpolate(x_list[i], size=(x0_h, x0_w), mode='bilinear', align_corners=False))
        x = torch.cat(x_all, dim=1)
        
        # Финальный слой
        x = self.final_layer(x)
        
        # Глобальный пулинг
        x = self.global_pool(x)
        backbone_features = x.view(batch_size, -1)
        
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
