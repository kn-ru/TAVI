import os
import yaml
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # Используем Agg бэкенд для работы без GUI
import matplotlib.pyplot as plt

from dataset import TAVIDataset, all_keypoint_classes
from hrnet_model import HRNetKeypointModel
from losses import KeypointLoss
from visualization import create_batch_visualization

def train(config):
    # Создаем директории для сохранения результатов
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    if config['visualization']['save_batch_images']:
        os.makedirs(config['visualization']['output_dir'], exist_ok=True)
    
    # Инициализируем датасеты и загрузчики данных
    # Используем предварительно разделенный датасет 512x512
    train_dataset = TAVIDataset(
        config['dataset_path'],
        mode='train',
        transform=True
    )
    
    val_dataset = TAVIDataset(
        config['dataset_path'],
        mode='val',
        transform=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Инициализируем модель HRNet
    model = HRNetKeypointModel(width=config.get('hrnet_width', 32))
    
    # Проверяем доступность GPU и достаточность памяти
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Используется устройство: {device}")
    
    # Если используется CUDA, выводим информацию о доступной памяти
    if use_cuda:
        print(f"Доступная память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model = model.to(device)
    
    # Инициализируем функцию потерь
    criterion = KeypointLoss(
        lambda_coord=config['lambda_coord'],
        lambda_group=config['lambda_group'],
        use_wing_loss=config.get('use_wing_loss', False)
    )
    
    # Инициализируем оптимизатор и планировщик скорости обучения
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Инициализируем TensorBoard для логирования
    writer = SummaryWriter()
    
    # Переменные для отслеживания лучшей модели
    best_val_loss = float('inf')
    
    # Основной цикл обучения
    for epoch in range(1, config['epochs'] + 1):
        epoch_start_time = time.time()
        # Обучение
        model.train()
        train_loss = 0.0
        train_presence_loss = 0.0
        train_coord_loss = 0.0
        train_group_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Теперь batch_data содержит только два элемента: изображения и ключевые точки
            if len(batch_data) == 3:  # Старый формат с метками групп
                images, keypoints, group_labels = batch_data
            else:  # Новый формат без меток групп
                images, keypoints = batch_data
                # Создаем метки групп на основе наличия точек
                batch_size = keypoints.size(0)
                group_labels = torch.zeros(batch_size, 3, dtype=torch.float32)
                
                # Заполняем метки групп на основе наличия точек в каждой группе
                # Используем фиксированные значения для индексов групп
                group1_size = 1  # Размер группы 1 (CP)
                group2_size = 6  # Размер группы 2 (FE2_o, FE1_o, FE2, FE1, EC1, EC2)
                group3_size = 3  # Размер группы 3 (CT1, CT2, CD)
                
                group1_indices = list(range(group1_size))
                group2_indices = list(range(group1_size, group1_size + group2_size))
                group3_indices = list(range(group1_size + group2_size, group1_size + group2_size + group3_size))
                
                for i in range(batch_size):
                    # Группа 1 присутствует, если хотя бы одна точка из группы 1 присутствует
                    group_labels[i, 0] = 1.0 if torch.any(keypoints[i, group1_indices, 0] > 0) else 0.0
                    # Группа 2 присутствует, если хотя бы одна точка из группы 2 присутствует
                    group_labels[i, 1] = 1.0 if torch.any(keypoints[i, group2_indices, 0] > 0) else 0.0
                    # Группа 3 присутствует, если хотя бы одна точка из группы 3 присутствует
                    group_labels[i, 2] = 1.0 if torch.any(keypoints[i, group3_indices, 0] > 0) else 0.0
            
            images = images.to(device)
            keypoints = keypoints.to(device)
            group_labels = group_labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)  # Словарь с выходами модели
            
            loss_dict = criterion(outputs, keypoints, group_labels)
            loss = loss_dict['total']
            loss_components = {
                'presence_loss': loss_dict['presence'].item(),
                'coord_loss': loss_dict['coord'].item(),
                'group_loss': loss_dict['group'].item()
            }
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_presence_loss += loss_components['presence_loss']
            train_coord_loss += loss_components['coord_loss']
            train_group_loss += loss_components['group_loss']
            
            # Выводим информацию о процессе обучения реже, чтобы не засорять логи
            if (batch_idx + 1) % 100 == 0:
                print(f"Эпоха {epoch}/{config['epochs']} [{batch_idx+1}/{len(train_loader)}] "
                      f"Потеря: {loss.item():.4f} "
                      f"(П: {loss_components['presence_loss']:.4f}, "
                      f"К: {loss_components['coord_loss']:.4f}, "
                      f"Г: {loss_components['group_loss']:.4f})")
            
            # Сохраняем визуализацию первого батча в каждой эпохе
            if batch_idx == 0 and config['visualization']['save_batch_images']:
                vis_path = os.path.join(config['visualization']['output_dir'], f'epoch_{epoch}_hrnet.png')
                batch_vis = create_batch_visualization(
                    images.cpu(),
                    keypoints.cpu(),
                    outputs['keypoints'].detach().cpu(),
                    save_path=vis_path,
                    max_images=config['visualization']['max_images']
                )
                # Файл уже сохранен в функции create_batch_visualization
                
                # Добавляем изображение в TensorBoard, если файл существует
                if os.path.exists(vis_path):
                    # Загружаем изображение и преобразуем его в формат для TensorBoard
                    img = plt.imread(vis_path)
                    writer.add_image('Batch Visualization', img.transpose(2, 0, 1), epoch)
        
        # Вычисляем средние потери за эпоху
        train_loss /= len(train_loader)
        train_presence_loss /= len(train_loader)
        train_coord_loss /= len(train_loader)
        train_group_loss /= len(train_loader)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        val_presence_loss = 0.0
        val_coord_loss = 0.0
        val_group_loss = 0.0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Теперь batch_data содержит только два элемента: изображения и ключевые точки
                if len(batch_data) == 3:  # Старый формат с метками групп
                    images, keypoints, group_labels = batch_data
                else:  # Новый формат без меток групп
                    images, keypoints = batch_data
                    # Создаем метки групп на основе наличия точек
                    batch_size = keypoints.size(0)
                    group_labels = torch.zeros(batch_size, 3, dtype=torch.float32)
                    
                    # Заполняем метки групп на основе наличия точек в каждой группе
                    # Используем фиксированные значения для индексов групп
                    group1_size = 1  # Размер группы 1 (CP)
                    group2_size = 6  # Размер группы 2 (FE2_o, FE1_o, FE2, FE1, EC1, EC2)
                    group3_size = 3  # Размер группы 3 (CT1, CT2, CD)
                    
                    group1_indices = list(range(group1_size))
                    group2_indices = list(range(group1_size, group1_size + group2_size))
                    group3_indices = list(range(group1_size + group2_size, group1_size + group2_size + group3_size))
                    
                    for i in range(batch_size):
                        # Группа 1 присутствует, если хотя бы одна точка из группы 1 присутствует
                        group_labels[i, 0] = 1.0 if torch.any(keypoints[i, group1_indices, 0] > 0) else 0.0
                        # Группа 2 присутствует, если хотя бы одна точка из группы 2 присутствует
                        group_labels[i, 1] = 1.0 if torch.any(keypoints[i, group2_indices, 0] > 0) else 0.0
                        # Группа 3 присутствует, если хотя бы одна точка из группы 3 присутствует
                        group_labels[i, 2] = 1.0 if torch.any(keypoints[i, group3_indices, 0] > 0) else 0.0
                
                images = images.to(device)
                keypoints = keypoints.to(device)
                group_labels = group_labels.to(device)
                
                outputs = model(images)
                
                loss_dict = criterion(outputs, keypoints, group_labels)
                loss = loss_dict['total']
                loss_components = {
                    'presence_loss': loss_dict['presence'].item(),
                    'coord_loss': loss_dict['coord'].item(),
                    'group_loss': loss_dict['group'].item()
                }
                
                val_loss += loss.item()
                val_presence_loss += loss_components['presence_loss']
                val_coord_loss += loss_components['coord_loss']
                val_group_loss += loss_components['group_loss']
        
        # Вычисляем средние потери за эпоху
        val_loss /= len(val_loader)
        val_presence_loss /= len(val_loader)
        val_coord_loss /= len(val_loader)
        val_group_loss /= len(val_loader)
        
        # Вычисляем время выполнения эпохи
        epoch_time = time.time() - epoch_start_time
        
        # Обновляем планировщик скорости обучения
        scheduler.step(val_loss)
        
        # Логируем потери в TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Loss/Train_Presence', train_presence_loss, epoch)
        writer.add_scalar('Loss/Train_Coord', train_coord_loss, epoch)
        writer.add_scalar('Loss/Train_Group', train_group_loss, epoch)
        writer.add_scalar('Loss/Val_Presence', val_presence_loss, epoch)
        writer.add_scalar('Loss/Val_Coord', val_coord_loss, epoch)
        writer.add_scalar('Loss/Val_Group', val_group_loss, epoch)
        
        # Выводим краткую информацию о результатах эпохи
        print(f"Эпоха {epoch}/{config['epochs']} | Время: {epoch_time:.1f}с | Обучение: {train_loss:.4f} | Валидация: {val_loss:.4f}")
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_hrnet_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Сохранена лучшая модель")
    
    # Закрываем TensorBoard writer
    writer.close()
    
    print("Обучение завершено!")

if __name__ == "__main__":
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Добавляем параметры для HRNet, если их нет в конфиге
    if 'hrnet_width' not in config:
        config['hrnet_width'] = 18  # Уменьшаем ширину каналов для экономии памяти
    
    # Проверяем наличие параметров визуализации
    if 'visualization' not in config:
        config['visualization'] = {
            'save_batch_images': True,
            'output_dir': 'visualizations',
            'max_images': 4
        }
    
    print(f"Используемая ширина HRNet: {config['hrnet_width']}")
    print(f"Размер батча: {config['batch_size']}")
    
    train(config)
