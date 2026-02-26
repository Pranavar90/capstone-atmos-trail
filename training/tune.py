import optuna
import torch
from training.train import DehazeDataset, DehazeTrainer
from torch.utils.data import DataLoader
from torchvision import transforms
import os

def objective(trial):
    config = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [2, 4, 8]),
        'physics_loss_weight': trial.suggest_float('physics_loss_weight', 0.1, 0.5),
        'perceptual_loss_weight': trial.suggest_float('perceptual_loss_weight', 0.05, 0.2),
        'use_mixed_precision': True,
        'grad_accum_steps': 2
    }
    
    image_size = 256
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    train_ds = DehazeDataset("data/processed", split="train", transform=transform)
    val_ds = DehazeDataset("data/processed", split="val", transform=transform)
    
    # Subsample for faster tuning
    train_indices = torch.randperm(len(train_ds))[:500]
    val_indices = torch.randperm(len(val_ds))[:100]
    
    train_sub = torch.utils.data.Subset(train_ds, train_indices)
    val_sub = torch.utils.data.Subset(val_ds, val_indices)
    
    train_loader = DataLoader(train_sub, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=config['batch_size'], shuffle=False)
    
    trainer = DehazeTrainer(config)
    
    # Run for a few epochs for each trial
    for epoch in range(5):
        trainer.train_epoch(train_loader)
        psnr = trainer.validate(val_loader)
        trial.report(psnr, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return psnr

def run_tuning():
    if not os.path.exists("data/processed/train/hazy"):
        print("Data not processed yet.")
        return

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Save best params
    with open("outputs/checkpoints/best_params.txt", "w") as f:
        f.write(str(trial.params))

if __name__ == "__main__":
    run_tuning()
