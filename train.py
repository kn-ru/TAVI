import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
import time
import numpy as np
import matplotlib
# Используем Agg бэкенд, который не требует графического интерфейса
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import GroupKeypointModel
from dataset import TAVIDataset, num_keypoints, group_1, group_2, group_3
from losses import KeypointLoss
from torch.utils.tensorboard import SummaryWriter
from visualization import create_batch_visualization

def train(config):
    # Инициализируем TensorBoard writer
    writer = SummaryWriter(f"runs/{config.get('experiment_name', 'keypoint_detection')}")
    
    # Создаем директорию для визуализаций, если она включена
    if config.get('visualization', {}).get('save_batch_images', False):
        vis_dir = config.get('visualization', {}).get('output_dir', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
    
    # Создаем датасет (путь к датасету указан в config)
    train_dataset = TAVIDataset(config['dataset_path'], mode='train')
    val_dataset = TAVIDataset(config['dataset_path'], mode='val')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Создаем модель с независимыми головами для групп
    model = GroupKeypointModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    model.to(device)

    # Оптимизатор и функции потерь
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Создаем комбинированную функцию потерь
    criterion = KeypointLoss(
        lambda_coord=config['lambda_coord'],
        lambda_group=config['lambda_group'],
        use_wing_loss=config.get('use_wing_loss', True)
    )
    
    # Создаем директорию для сохранения моделей
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    best_val_loss = float('inf')
    
    # Обучение модели
    for epoch in range(config['epochs']):
        # Обучение
        model.train()
        train_loss = 0
        train_presence_loss = 0
        train_coords_loss = 0
        train_group_loss = 0
        
        # Флаг для отслеживания первого батча в эпохе
        first_batch = True
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            images, keypoints = batch
            images = images.to(device)
            keypoints = keypoints.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # Словарь с выходами модели

            # Вычисляем потери с помощью комбинированной функции потерь
            losses = criterion(outputs, keypoints)
            
            # Получаем общую потерю и её компоненты
            loss = losses['total']
            loss_presence = losses['presence']
            loss_coords = losses['coord']
            loss_group = losses['group']

            # Обратное распространение и оптимизация
            loss.backward()
            optimizer.step()

            # Накапливаем статистику
            train_loss += loss.item()
            train_presence_loss += loss_presence.item()
            train_coords_loss += loss_coords.item()
            train_group_loss += loss_group.item()
            
            # Логируем метрики в TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/total_loss', loss.item(), global_step)
            writer.add_scalar('train/presence_loss', loss_presence.item(), global_step)
            writer.add_scalar('train/coord_loss', loss_coords.item(), global_step)
            writer.add_scalar('train/group_loss', loss_group.item(), global_step)
            
            # Сохраняем визуализацию первого батча в каждой эпохе
            if first_batch and config.get('visualization', {}).get('save_batch_images', False):
                vis_dir = config.get('visualization', {}).get('output_dir', 'visualizations')
                max_images = config.get('visualization', {}).get('max_images', 16)
                
                # Создаем визуализацию батча
                vis_path = os.path.join(vis_dir, f'batch_epoch_{epoch+1}.png')
                create_batch_visualization(images.detach(), keypoints.detach(), outputs, vis_path, max_images)
                
                # Добавляем визуализацию в TensorBoard
                image_data = plt.imread(vis_path)
                writer.add_image('train/batch_visualization', image_data, epoch, dataformats='HWC')
                
                # Сбрасываем флаг
                first_batch = False
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Эпоха {epoch+1}/{config['epochs']} [{batch_idx+1}/{len(train_loader)}] "
                      f"Потеря: {loss.item():.4f} (П: {loss_presence.item():.4f}, К: {loss_coords.item():.4f}, Г: {loss_group.item():.4f})")
        
        # Вычисляем средние потери за эпоху
        train_loss /= len(train_loader)
        train_presence_loss /= len(train_loader)
        train_coords_loss /= len(train_loader)
        train_group_loss /= len(train_loader)
        
        # Валидация
        model.eval()
        val_loss = 0
        val_presence_loss = 0
        val_coords_loss = 0
        val_group_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images, keypoints = batch
                images = images.to(device)
                keypoints = keypoints.to(device)

                # Получаем выходы модели
                outputs = model(images)

                # Вычисляем потери с помощью комбинированной функции потерь
                losses = criterion(outputs, keypoints)
                
                # Получаем общую потерю и её компоненты
                loss = losses['total']
                loss_presence = losses['presence']
                loss_coords = losses['coord']
                loss_group = losses['group']

                # Накапливаем статистику
                val_loss += loss.item()
                val_presence_loss += loss_presence.item()
                val_coords_loss += loss_coords.item()
                val_group_loss += loss_group.item()
        
        # Вычисляем средние потери за эпоху
        val_loss /= len(val_loader)
        val_presence_loss /= len(val_loader)
        val_coords_loss /= len(val_loader)
        val_group_loss /= len(val_loader)
        
        # Обновляем learning rate на основе валидационной потери
        scheduler.step(val_loss)
        
        # Логируем метрики валидации в TensorBoard
        writer.add_scalar('val/total_loss', val_loss, epoch)
        writer.add_scalar('val/presence_loss', val_presence_loss, epoch)
        writer.add_scalar('val/coord_loss', val_coords_loss, epoch)
        writer.add_scalar('val/group_loss', val_group_loss, epoch)
        
        # Выводим статистику
        epoch_time = time.time() - start_time
        print(f"Эпоха {epoch+1}/{config['epochs']} завершена за {epoch_time:.2f}s")
        print(f"Потеря на обучении: {train_loss:.4f} (П: {train_presence_loss:.4f}, К: {train_coords_loss:.4f}, Г: {train_group_loss:.4f})")
        print(f"Потеря на валидации: {val_loss:.4f} (П: {val_presence_loss:.4f}, К: {val_coords_loss:.4f}, Г: {val_group_loss:.4f})")
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'val_coords_loss': val_coords_loss,
                'config': config
            }, checkpoint_path)
            print(f"Сохранена лучшая модель с потерей на валидации: {val_loss:.4f}")
        
        # Сохраняем последнюю модель
        checkpoint_path = os.path.join(config['checkpoint_dir'], 'last_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'val_coords_loss': val_coords_loss,
            'config': config
        }, checkpoint_path)
    
    print("Training completed!")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train(config)
