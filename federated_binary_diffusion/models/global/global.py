import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import List, Dict

class FederatedServer:
    def __init__(self, model_config, diffusion_config):
        self.model = SimpleTableGenerator.from_config(model_config)
        self.diffusion = BinaryDiffusion1D.from_config(
            denoise_model=self.model,
            config=diffusion_config
        )
        
    def aggregate_models(self, client_models: List[Dict[str, torch.Tensor]]) -> None:
        """FedAvg algorithm to aggregate client models"""
        averaged_weights = {}
        
        # Get the first client's weights as starting point
        for key in client_models[0].keys():
            averaged_weights[key] = torch.zeros_like(client_models[0][key])
            
        # Average the weights
        for key in averaged_weights.keys():
            for client_weight in client_models:
                averaged_weights[key] += client_weight[key]
            averaged_weights[key] = torch.div(averaged_weights[key], len(client_models))
            
        # Update the global model
        self.diffusion.load_state_dict(averaged_weights)
        
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """Return the current global model weights"""
        return deepcopy(self.diffusion.state_dict())

class FederatedClient:
    def __init__(self, data: pd.DataFrame, model_config, diffusion_config):
        self.dataset = self.prepare_dataset(data)
        self.model = SimpleTableGenerator.from_config(model_config)
        self.diffusion = BinaryDiffusion1D.from_config(
            denoise_model=self.model,
            config=diffusion_config
        )
        self.trainer = None
        
    def prepare_dataset(self, data: pd.DataFrame) -> FixedSizeBinaryTableDataset:
        """Prepare the dataset using your existing configuration"""
        # Implement dataset preparation based on your current code
        pass
        
    def update_local_model(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Update local model with global weights"""
        self.diffusion.load_state_dict(global_weights)
        
    def train_local_model(self, num_steps: int) -> Dict[str, torch.Tensor]:
        """Train the local model for specified number of steps"""
        if self.trainer is None:
            self.trainer = FixedSizeTableBinaryDiffusionTrainer(
                diffusion=self.diffusion,
                dataset=self.dataset,
                train_num_steps=num_steps,
                # Add other trainer parameters from your config
            )
        
        self.trainer.train()
        return self.diffusion.state_dict()

def federated_training(
    server: FederatedServer,
    clients: List[FederatedClient],
    num_rounds: int,
    local_steps: int
) -> Dict:
    """Run federated training process"""
    metrics = {
        'rounds': [],
        'client_losses': [],
        'global_metrics': []
    }
    
    for round_idx in range(num_rounds):
        print(f"Federated Training Round {round_idx + 1}/{num_rounds}")
        
        # Get current global model
        global_weights = server.get_global_model()
        
        # Train each client locally
        client_weights = []
        client_round_losses = []
        
        for client_idx, client in enumerate(clients):
            print(f"Training Client {client_idx + 1}/{len(clients)}")
            client.update_local_model(global_weights)
            local_weights = client.train_local_model(local_steps)
            client_weights.append(local_weights)
            # Collect metrics if needed
            
        # Aggregate models
        server.aggregate_models(client_weights)
        
        # Save metrics
        metrics['rounds'].append(round_idx)
        metrics['client_losses'].append(client_round_losses)
        # Add any global metrics you want to track
        
    return metrics
