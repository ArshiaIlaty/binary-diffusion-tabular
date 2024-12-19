import os
import sys
import torch
import yaml
from pathlib import Path

# Add binary-diffusion-tabular to Python path
binary_diffusion_path = str(Path('../binary-diffusion-tabular').resolve())
sys.path.append(binary_diffusion_path)

from utils import (
    FederatedLogger,
    compare_data_distributions,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_model_comparison,
    save_metrics_report
)

from binary_diffusion_tabular import (
    FixedSizeBinaryTableDataset,
    SimpleTableGenerator,
    BinaryDiffusion1D,
    FixedSizeTableBinaryDiffusionTrainer
)

class FederatedTraining:
    def __init__(self, client1_config, client2_config):
        self.client1_config = self._load_config(client1_config)
        self.client2_config = self._load_config(client2_config)
        self.global_model = None

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _init_global_model(self):
        """Initialize global model with same architecture as clients"""
        self.global_model = SimpleTableGenerator.from_config(
            self.client1_config['model']
        ).to('cuda')
        
    def _train_client(self, config, round_num):
        """Train a client model for one federated round"""
        trainer = FixedSizeTableBinaryDiffusionTrainer.from_config(
            config,
            logger=None  # Add wandb logger if needed
        )
        
        if self.global_model is not None:
            trainer.diffusion.load_state_dict(self.global_model.state_dict())
            
        trainer.train()
        return trainer.diffusion
    
    def run_federated_training(self, num_rounds=10):
        """Run federated training for specified number of rounds"""
        for round_num in range(num_rounds):
            print(f"\nFederated Round {round_num + 1}/{num_rounds}")
            
            # Train client 1
            print("Training Client 1...")
            client1_model = self._train_client(self.client1_config, round_num)
            
            # Train client 2
            print("Training Client 2...")
            client2_model = self._train_client(self.client2_config, round_num)
            
            # Average models (FedAvg)
            if self.global_model is None:
                self._init_global_model()
                
            # Perform model averaging
            self._average_models([client1_model, client2_model])
            
            # Save global model
            self._save_global_model(round_num)
    
    def _average_models(self, client_models):
        """Implement FedAvg algorithm"""
        with torch.no_grad():
            for param in self.global_model.parameters():
                param.data.zero_()
                
            for client_model in client_models:
                for global_param, client_param in zip(
                    self.global_model.parameters(),
                    client_model.parameters()
                ):
                    global_param.data.add_(client_param.data / len(client_models))
    
    def _save_global_model(self, round_num):
        """Save the global model after each round"""
        save_path = f"./models/global/global_model_round_{round_num}.pt"
        torch.save(self.global_model.state_dict(), save_path)

if __name__ == "__main__":
    federated = FederatedTraining(
        client1_config="./configs/client1_config.yaml",
        client2_config="./configs/client2_config.yaml"
    )
    federated.run_federated_training(num_rounds=10)
