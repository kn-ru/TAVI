import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')  # Используем Agg бэкенд для работы без GUI

from dataset import KeypointDataset
from hrnet_model import HRNetKeypointModel
from losses import KeypointLoss
from visualization import create_batch_visualization

def train(config):
    # Создаем директории для сохранения результатов
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    if config['visualization']['save_batch_images']:
        os.makedirs(config['visualization']['output_dir'], exist_ok=True)
    
    # Инициализируем датасеты и загрузчики данных
    train_dataset = KeypointDataset(
        config['train_dir'],
        augmentation=True,
        img_size=config['img_size']
    )
    
    val_dataset = KeypointDataset(
        config['val_dir'],
        augmentation=False,
        img_size=config['img_size']
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
    
    # Определяем устройство (GPU или CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
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
        # Обучение
        model.train()
        train_loss = 0.0
        train_presence_loss = 0.0
        train_coord_loss = 0.0
        train_group_loss = 0.0
        
        for batch_idx, (images, keypoints, group_labels) in enumerate(train_loader):
            images = images.to(device)
            keypoints = keypoints.to(device)
            group_labels = group_labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)  # Словарь с выходами модели
            
            loss, loss_components = criterion(outputs, keypoints, group_labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_presence_loss += loss_components['presence_loss']
            train_coord_loss += loss_components['coord_loss']
            train_group_loss += loss_components['group_loss']
            
            # Выводим информацию о процессе обучения
            if (batch_idx + 1) % 20 == 0:
                print(f"Эпоха {epoch}/{config['epochs']} [{batch_idx+1}/{len(train_loader)}] "
                      f"Потеря: {loss.item():.4f} "
                      f"(П: {loss_components['presence_loss']:.4f}, "
                      f"К: {loss_components['coord_loss']:.4f}, "
                      f"Г: {loss_components['group_loss']:.4f})")
            
            # Сохраняем визуализацию первого батча в каждой эпохе
            if batch_idx == 0 and config['visualization']['save_batch_images']:
                batch_vis = create_batch_visualization(
                    images.cpu(),
                    keypoints.cpu(),
                    outputs['keypoints'].detach().cpu(),
                    max_images=config['visualization']['max_images']
                )
                vis_path = os.path.join(config['visualization']['output_dir'], f'epoch_{epoch}_hrnet.png')
                batch_vis.savefig(vis_path)
                
                # Добавляем изображение в TensorBoard
                writer.add_figure('Batch Visualization', batch_vis, epoch)
        
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
            for images, keypoints, group_labels in val_loader:
                images = images.to(device)
                keypoints = keypoints.to(device)
                group_labels = group_labels.to(device)
                
                outputs = model(images)
                
                loss, loss_components = criterion(outputs, keypoints, group_labels)
                
                val_loss += loss.item()
                val_presence_loss += loss_components['presence_loss']
                val_coord_loss += loss_components['coord_loss']
                val_group_loss += loss_components['group_loss']
        
        # Вычисляем средние потери за эпоху
        val_loss /= len(val_loader)
        val_presence_loss /= len(val_loader)
        val_coord_loss /= len(val_loader)
        val_group_loss /= len(val_loader)
        
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
        
        # Выводим информацию о потерях
        print(f"Эпоха {epoch}/{config['epochs']} завершена за {epoch_time:.2f}s")
        print(f"Потеря на обучении: {train_loss:.4f} "
              f"(П: {train_presence_loss:.4f}, "
              f"К: {train_coord_loss:.4f}, "
              f"Г: {train_group_loss:.4f})")
        print(f"Потеря на валидации: {val_loss:.4f} "
              f"(П: {val_presence_loss:.4f}, "
              f"К: {val_coord_loss:.4f}, "
              f"Г: {val_group_loss:.4f})")
        
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
            print(f"Сохранена лучшая модель с потерей на валидации: {val_loss:.4f}")
    
    # Закрываем TensorBoard writer
    writer.close()
    
    print("Обучение завершено!")

if __name__ == "__main__":
    # Загружаем конфигурацию
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Добавляем параметры для HRNet
    config['hrnet_width'] = 32  # Ширина каналов в HRNet
    
    train(config)
