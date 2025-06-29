import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Hiperparâmetros e configurações do projeto CNN.")

    # Wandb
    parser.add_argument("--wandb_name_project", type=str)
    parser.add_argument("--wandb_entidade", type=str)
    parser.add_argument("--wandb_sweep", action="store_true")

    # Kaggle
    parser.add_argument("--usuario_kaggle", type=str)
    parser.add_argument("--api_kaggle", type=str)
    parser.add_argument("--dataset_id", type=str)
    parser.add_argument("--force_download", action="store_true")

    # Hiperparâmetros
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--activation", type=str, choices=['relu', 'selu', 'mish'], default='relu')
    parser.add_argument("--dense_layer", type=int, default=128)
    parser.add_argument("--congelar_rede", action="store_true")
    parser.add_argument("--batch_normalisation", action="store_true")
    parser.add_argument("--data_augmentation", action="store_true")

    return parser.parse_args()

