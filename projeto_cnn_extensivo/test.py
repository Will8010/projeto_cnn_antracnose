import torch
from model import ConvNeuNet
from preprocess import data_pre_processing
from utils import evaluate

def test_model(args):
    print("Testando modelo com os melhores hiperparâmetros...")

    model = ConvNeuNet(
        size_kernel=[(3,3)]*5,
        num_stride=1,
        act_fu='selu',
        size_denseLayer=200,
        data_augmentation=True,
        batch_normalisation=True,
        padding=1,
        dropout_rate=0.3,
        num_filters=[12]*5,
        classes=10,
        input_channels=3
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = data_pre_processing(
        train_path="path/para/train",
        test_path="path/para/test",
        batch_size=32,
        data_augmentation=True
    )

    model.load_state_dict(torch.load("best_model.pth"))
    accuracy = evaluate(model, test_loader)
    print(f"Acurácia no conjunto de teste: {accuracy:.2f}%")