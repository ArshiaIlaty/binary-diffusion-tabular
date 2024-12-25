from binary_diffusion_tabular import (
    FixedSizeBinaryTableDataset,
    SimpleTableGenerator,
    BinaryDiffusion1D,
    FixedSizeTableBinaryDiffusionTrainer
)
from utils import (
    FederatedLogger,
    compare_data_distributions,
    plot_roc_curves,
    plot_confusion_matrices,
    plot_model_comparison,
    save_metrics_report
)
import os
import sys
import torch
import yaml
import argparse
from pathlib import Path

# Add binary-diffusion-tabular to Python path
binary_diffusion_path = str(Path('../binary-diffusion-tabular').resolve())
sys.path.append(binary_diffusion_path)

class FederatedTraining:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.global_model = None
        self.model_params = None
        self._init_model_params()
        
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _init_model_params(self):
        """Initialize model parameters from config"""
        # Create dataset for architecture reference
        dataset = FixedSizeBinaryTableDataset.from_config(self.config['data'])
        
        # Store architecture parameters
        self.model_params = {
            'data_dim': dataset.row_size,
            'dim': self.config['model']['dim'],
            'n_res_blocks': self.config['model']['n_res_blocks'],
            'out_dim': (
                dataset.row_size * 2
                if self.config['diffusion']['target'] == "two_way"
                else dataset.row_size
            ),
            'task': self.config['data']['task'],
            'conditional': dataset.conditional,
            'n_classes': 0 if self.config['data']['task'] == "regression" else dataset.n_classes,
            'classifier_free_guidance': self.config['trainer']['classifier_free_guidance']
        }

    def _split_data_for_clients(self):
        """Split the dataset for two clients"""
        dataset = FixedSizeBinaryTableDataset.from_config(self.config['data'])
        total_size = len(dataset)
        split_idx = total_size // 2
        
        # Create configs for each client
        self.client1_config = self.config.copy()
        self.client2_config = self.config.copy()
        
        # Modify data paths or indices for each client
        # This depends on how your data is structured
        # For example, you might want to split the data file or modify data indices
        
        return self.client1_config, self.client2_config

    def _train_client(self, config, round_num):
        """Train a client model for one federated round"""
        # Initialize model with current parameters
        model = SimpleTableGenerator(**self.model_params).to('cuda')
        
        # If global model exists, load its state
        if self.global_model is not None:
            model.load_state_dict(self.global_model.state_dict())
        
        # Initialize trainer
        trainer = FixedSizeTableBinaryDiffusionTrainer(
            diffusion=model,
            config=config,
            logger=None  # Add wandb logger if needed
        )
        
        # Train the model
        trainer.train()
        return trainer.diffusion

    def _init_global_model(self):
        """Initialize the global model"""
        self.global_model = SimpleTableGenerator(**self.model_params).to('cuda')

    def run_federated_training(self, num_rounds=10):
        """Run federated training for specified number of rounds"""
        # Initialize global model if not done
        if self.global_model is None:
            self._init_global_model()
            
        # Split data for clients
        client1_config, client2_config = self._split_data_for_clients()
        
        for round_num in range(num_rounds):
            print(f"\nFederated Round {round_num + 1}/{num_rounds}")
            
            # Train client 1
            print("Training Client 1...")
            client1_model = self._train_client(client1_config, round_num)
            
            # Train client 2
            print("Training Client 2...")
            client2_model = self._train_client(client2_config, round_num)
            
            # Average models (FedAvg)
            self._average_models([client1_model, client2_model])
            
            # Save global model
            self._save_global_model(round_num)

    def _average_models(self, client_models):
        """Implement FedAvg algorithm"""
        with torch.no_grad():
            # Zero out global model parameters
            for param in self.global_model.parameters():
                param.data.zero_()
            
            # Average the parameters
            for client_model in client_models:
                for global_param, client_param in zip(
                    self.global_model.parameters(),
                    client_model.parameters()
                ):
                    global_param.data.add_(client_param.data / len(client_models))

    def _save_global_model(self, round_num):
        """Save the global model after each round"""
        os.makedirs("./models/global", exist_ok=True)
        save_path = f"./models/global/global_model_round_{round_num}.pt"
        torch.save(self.global_model.state_dict(), save_path)

def main():
    parser = argparse.ArgumentParser(description='Federated Binary Diffusion Training')
    parser.add_argument('-c', '--config', type=str, required=True,
                      help='Path to configuration file')
    args = parser.parse_args()
    
    federated = FederatedTraining(args.config)
    federated.run_federated_training(num_rounds=10)

if __name__ == "__main__":
    main()