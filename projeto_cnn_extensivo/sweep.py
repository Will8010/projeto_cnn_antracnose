import wandb
from train import train_model

def sweep(args):
    sweep_config = {
        "name": args.wandb_name_project,
        "method": "bayes",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": {
            "num_filters": {"values": [[12,12,12,12,12],[4,8,16,32,64]]},
            "activation": {"values": ["relu", "selu", "mish"]},
            "dropout_rate": {"values": [0.2, 0.3, 0.5]},
            "dense_layer": {"values": [64, 128, 256]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, entity=args.wandb_entidade, project=args.wandb_name_project)
    wandb.agent(sweep_id, function=lambda: train_model(args, sweep=True))