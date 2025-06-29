import torch
from args_parser import get_args
from dataset import baixar_dataset_kaggle
from preprocess import data_pre_processing
from train import train_CNN

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = baixar_dataset_kaggle(
        usuario_kaggle=args.usuario_kaggle,
        api_kaggle=args.api_kaggle,
        dataset_id=args.dataset_id,
        force_download=args.force_download
    )

    train_path = f"{dataset_path}/train"
    test_path = f"{dataset_path}/val"

    train_loader, val_loader, test_loader = data_pre_processing(
        train_path=train_path,
        test_path=test_path,
        batch_size=args.batch_size,
        data_augmentation=args.data_augmentation
    )

    model = train_CNN(args, train_loader, val_loader, device)

if __name__ == "__main__":
    main()
