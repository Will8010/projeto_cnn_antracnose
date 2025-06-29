import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, ConcatDataset

def data_pre_processing(train_path, test_path, batch_size, data_augmentation):
    all_transforms = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4713, 0.4600, 0.3897], std=[0.2373, 0.2266, 0.2374])
    ])

    train_dataset = datasets.ImageFolder(root=train_path, transform=all_transforms)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    if data_augmentation:
        augment_transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-60, 60)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4713, 0.4600, 0.3897], std=[0.2373, 0.2266, 0.2374])
        ])
        aug_dataset = datasets.ImageFolder(root=train_path, transform=augment_transforms)
        _, transformed_dataset = random_split(aug_dataset, [int(0.8 * len(aug_dataset)), len(aug_dataset) - int(0.8 * len(aug_dataset))])
        train_dataset = ConcatDataset([train_dataset, transformed_dataset])

    test_dataset = datasets.ImageFolder(root=test_path, transform=all_transforms)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )