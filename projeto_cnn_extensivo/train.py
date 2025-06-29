import torch
import torch.nn as nn
from utils import evaluate
from model import ConvNeuNet
import wandb

def train_CNN(args, train_loader, val_loader, device):
    if args.wandb_sweep:
        wandb.init(project=args.wandb_name_project, entity=args.wandb_entidade)
        config = wandb.config
    else:
        config = args

    model = ConvNeuNet(
        size_kernel=[(3,3)]*5,
        num_stride=1,
        act_fu=config.activation,
        size_denseLayer=config.dense_layer,
        data_augmentation=config.data_augmentation,
        batch_normalisation=config.batch_normalisation,
        padding=1,
        dropout_rate=0.3,
        num_filters=[12]*5,
        input_channels=3,
        classes=10
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_acc = evaluate(model, train_loader, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}: Loss={running_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        if args.wandb_sweep:
            wandb.log({
                "loss": running_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "epoch": epoch+1
            })

    if not args.wandb_sweep:
        return model
