import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np

from models.st_gnn import STGNNDetector
from data.dataset import RadarDataset
from utils.config import get_config_value, load_config
from utils.metrics import compute_detection_metrics


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    total_batches = len(dataloader)

    for batch_idx, (E_real, E_imag, labels) in enumerate(dataloader):
        E_real = E_real.to(device)
        E_imag = E_imag.to(device)
        labels = labels.to(device).unsqueeze(1)

        E = torch.complex(E_real, E_imag)

        optimizer.zero_grad()
        outputs = model(E)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_predictions.extend(outputs.detach().cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())

        if (batch_idx + 1) % 10 == 0:
            progress = (batch_idx + 1) / total_batches * 100
            print(f"  Batch {batch_idx+1}/{total_batches} ({progress:.1f}%), Loss: {loss.item():.4f}")

    metrics = compute_detection_metrics(
        torch.tensor(all_predictions),
        torch.tensor(all_labels)
    )

    return total_loss / len(dataloader), metrics


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for E_real, E_imag, labels in dataloader:
            E_real = E_real.to(device)
            E_imag = E_imag.to(device)
            labels = labels.to(device).unsqueeze(1)

            E = torch.complex(E_real, E_imag)

            outputs = model(E)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_predictions.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    metrics = compute_detection_metrics(
        torch.tensor(all_predictions),
        torch.tensor(all_labels)
    )

    return total_loss / len(dataloader), metrics


def main():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(description='Train ST-GNN Detector')
    parser.add_argument("--config", type=str, default=config_args.config,
                        help="TOML config path. CLI options override config values.")
    parser.add_argument('--data_dir', type=str, default=get_config_value(config, "paths.data_dir"),
                        help='Data directory')
    parser.add_argument('--epochs', type=int, default=get_config_value(config, "train.epochs"),
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=get_config_value(config, "train.batch_size"),
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=get_config_value(config, "train.learning_rate"),
                        help='Learning rate')
    parser.add_argument('--P', type=int, default=get_config_value(config, "model.pulses"),
                        help='Number of pulses')
    parser.add_argument('--N', type=int, default=get_config_value(config, "model.range_cells"),
                        help='Number of range cells')
    parser.add_argument('--save_dir', type=str, default=get_config_value(config, "paths.save_dir"),
                        help='Save directory')
    parser.add_argument('--num_workers', type=int, default=get_config_value(config, "train.num_workers"),
                        help='DataLoader worker count')
    args = parser.parse_args(remaining)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"ST-GNN Detector Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    if args.config:
        print(f"Config: {args.config}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}")

    os.makedirs(args.save_dir, exist_ok=True)

    print("\nLoading datasets...")
    try:
        train_dataset = RadarDataset(args.data_dir, P=args.P, N=args.N, train=True)
        val_dataset = RadarDataset(args.data_dir, P=args.P, N=args.N, train=False)
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please generate simulated data first: python data/simulator.py")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("\nInitializing model...")
    model = STGNNDetector(P=args.P, N=args.N).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')

    print(f"\nStarting training...")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Train Precision: {train_metrics['precision']:.4f} | Recall: {train_metrics['recall']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved: {save_path}")

    final_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final model saved: {final_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
