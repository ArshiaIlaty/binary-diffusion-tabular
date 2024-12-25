import syft as sy
import torch
from binary_diffusion_tabular import (
    FixedSizeBinaryTableDataset,
    SimpleTableGenerator,
    FixedSizeTableBinaryDiffusionTrainer
)
import yaml

def create_virtual_workers():
    hook = sy.TorchHook(torch)
    client1 = sy.VirtualWorker(hook, id="client1")
    client2 = sy.VirtualWorker(hook, id="client2")
    return client1, client2

class FederatedBinaryDiffusion:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset = FixedSizeBinaryTableDataset.from_config(self.config['data'])
        self.model = self._init_model()
        self.client1, self.client2 = create_virtual_workers()

    def _init_model(self):
        model_params = {
            'data_dim': self.dataset.row_size,
            'dim': self.config['model']['dim'],
            'n_res_blocks': self.config['model']['n_res_blocks'],
            'out_dim': (
                self.dataset.row_size * 2
                if self.config['diffusion']['target'] == "two_way"
                else self.dataset.row_size
            ),
            'task': self.config['data']['task'],
            'conditional': self.dataset.conditional,
            'n_classes': 0 if self.config['data']['task'] == "regression" else self.dataset.n_classes,
            'classifier_free_guidance': self.config['trainer']['classifier_free_guidance']
        }
        return SimpleTableGenerator(**model_params)

    def split_and_send_data(self):
        # Split dataset between clients
        total_size = len(self.dataset)
        split_idx = total_size // 2
        data1 = self.dataset[:split_idx]
        data2 = self.dataset[split_idx:]
        
        # Send data to virtual workers
        data1_ptr = data1.send(self.client1)
        data2_ptr = data2.send(self.client2)
        return data1_ptr, data2_ptr

    def train_federated(self, num_rounds=10):
        # Send model to workers
        model_ptr1 = self.model.copy().send(self.client1)
        model_ptr2 = self.model.copy().send(self.client2)
        
        data1_ptr, data2_ptr = self.split_and_send_data()
        
        for round in range(num_rounds):
            print(f"Round {round + 1}/{num_rounds}")
            
            # Train on client 1
            trainer1 = FixedSizeTableBinaryDiffusionTrainer(
                diffusion=model_ptr1,
                config=self.config,
                logger=None
            )
            trainer1.train()
            
            # Train on client 2
            trainer2 = FixedSizeTableBinaryDiffusionTrainer(
                diffusion=model_ptr2,
                config=self.config,
                logger=None
            )
            trainer2.train()
            
            # Average models
            with torch.no_grad():
                model1 = model_ptr1.get()
                model2 = model_ptr2.get()
                
                # Perform FedAvg
                for param1, param2 in zip(model1.parameters(), model2.parameters()):
                    param1.data.add_(param2.data)
                    param1.data.div_(2.0)
                
                # Send updated model back to workers
                model_ptr1 = model1.send(self.client1)
                model_ptr2 = model1.send(self.client2)
            
            # Save global model
            self.save_model(round, model1)

    def save_model(self, round_num: int, model: torch.nn.Module):
        save_path = f"models/global_model_round_{round_num}.pt"
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--rounds', type=int, default=10, help='Number of federated rounds')
    args = parser.parse_args()
    
    fed_trainer = FederatedBinaryDiffusion(args.config)
    fed_trainer.train_federated(num_rounds=args.rounds)