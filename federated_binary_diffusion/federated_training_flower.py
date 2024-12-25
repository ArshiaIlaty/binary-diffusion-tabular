import os
import sys
import torch
import wandb
import yaml
import numpy as np
import flwr as fl
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add binary-diffusion-tabular to Python path
binary_diffusion_path = str(Path('../binary-diffusion-tabular').resolve())
sys.path.append(binary_diffusion_path)

try:
    from binary_diffusion_tabular import (
        FixedSizeBinaryTableDataset,
        SimpleTableGenerator,
        BinaryDiffusion1D,
        FixedSizeTableBinaryDiffusionTrainer
    )
except ImportError as e:
    logger.error(f"Failed to import binary_diffusion_tabular: {e}")
    raise

def cycle(iterable):
    """Creates an infinite iterator that cycles through the data."""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

class BinaryDiffusionClient(fl.client.NumPyClient):
    """
    Federated Learning client for binary diffusion models.
    Handles local training, parameter updates, and evaluation.
    """
    
    def __init__(self, config_path: str, client_id: str):
        """
        Initialize the Binary Diffusion Client.
        
        Args:
            config_path (str): Path to the configuration file
            client_id (str): Unique identifier for the client
        """
        logger.info(f"Initializing client {client_id}...")
        self.client_id = client_id
        
        try:
            self._initialize_wandb()
            self._load_config(config_path)
            self._setup_dataset()
            self._setup_model()
            self._setup_diffusion()
            self._setup_trainer()
            
            logger.info(f"Client {client_id} setup complete!")
            
        except Exception as e:
            logger.error(f"Error during client initialization: {str(e)}")
            raise

    def _initialize_wandb(self):
        """Initialize Weights & Biases logging."""
        logger.info("Starting W&B...")
        self.run = wandb.init(
            project="federated-binary-diffusion",
            entity="ilatyarshia",
            name=f"client-{self.client_id}",
            group="federated-training"
        )

    def _load_config(self, config_path: str):
        """Load configuration from YAML file."""
        logger.info("Loading config...")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        logger.info("Config loaded successfully")

    def _setup_dataset(self):
        """Initialize and set up the dataset."""
        logger.info("Initializing dataset...")
        self.dataset = FixedSizeBinaryTableDataset.from_config(self.config['data'])
        logger.info(f"Dataset initialized with {len(self.dataset)} samples")

    def _setup_model(self):
        """Initialize the model architecture."""
        logger.info("Initializing model...")
        
        # Calculate dimensions
        num_features = len(self.config['data']['numerical_columns'])
        if 'categorical_columns' in self.config['data']:
            num_features += len(self.config['data']['categorical_columns'])
        
        hidden_dim = self.config['model']['dim']
        output_dim = num_features * 2 if self.config['diffusion']['target'] == "two_way" else num_features
        
        self.model = SimpleTableGenerator(
            data_dim=num_features,
            dim=hidden_dim,
            n_res_blocks=self.config['model']['n_res_blocks'],
            out_dim=output_dim,
            task=self.config['data']['task'],
            conditional=True,
            n_classes=2 if self.config['data']['task'] == 'classification' else 0,
            classifier_free_guidance=self.config['trainer'].get('classifier_free_guidance', True)
        ).to('cuda')
        
        logger.info(f"Model architecture:")
        logger.info(f"- Input dimension: {num_features}")
        logger.info(f"- Hidden dimension: {hidden_dim}")
        logger.info(f"- Output dimension: {output_dim}")

    def _setup_diffusion(self):
        """Initialize the diffusion process."""
        logger.info("Initializing diffusion...")
        self.diffusion = BinaryDiffusion1D(
            self.model,
            n_timesteps=self.config['diffusion']['n_timesteps']
        )
        logger.info("Diffusion initialized")

    def _setup_trainer(self):
        """Initialize the trainer with custom dataloader."""
        logger.info("Initializing trainer...")
        
        # Create results directory
        results_folder = self.config.get('results_folder', f'./models/{self.client_id}')
        os.makedirs(results_folder, exist_ok=True)
        
        # Setup data loader
        self.train_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config['trainer'].get('batch_size', 32),
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # Initialize trainer
        self.trainer = FixedSizeTableBinaryDiffusionTrainer(
            diffusion=self.diffusion,
            dataset=self.dataset,
            batch_size=self.config['trainer'].get('batch_size', 32),
            train_num_steps=self.config['trainer'].get('train_num_steps', 1000),
            gradient_accumulate_every=self.config['trainer'].get('gradient_accumulate_every', 1),
            ema_decay=self.config['trainer'].get('ema_decay', 0.995),
            ema_update_every=self.config['trainer'].get('ema_update_every', 10),
            save_every=self.config['trainer'].get('save_every', 1000),
            log_every=self.config['trainer'].get('log_every', 100),
            logger=self.run,
            opt_type=self.config['trainer'].get('opt_type', 'adam'),
            lr=self.config['trainer'].get('learning_rate', 0.001),
            classifier_free_guidance=True,
            zero_token_probability=self.config['trainer'].get('zero_token_probability', 0.1),
            results_folder=results_folder
        )
        
        # Set up cycling dataloader
        self.trainer.dataloader = cycle(self.train_loader)

    def get_parameters(self, config) -> List[np.ndarray]:
        """
        Get model parameters.
        
        Returns:
            List[np.ndarray]: List of model parameters as numpy arrays
        """
        try:
            params = [val.cpu().numpy() for _, val in self.diffusion.model.state_dict().items()]
            logger.info(f"Parameter shapes: {[p.shape for p in params]}")
            return params
        except Exception as e:
            logger.error(f"Error getting parameters: {str(e)}")
            raise

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters.
        
        Args:
            parameters (List[np.ndarray]): List of model parameters as numpy arrays
        """
        try:
            logger.info(f"Received parameter shapes: {[p.shape for p in parameters]}")
            logger.info(f"Current model state dict keys: {self.diffusion.model.state_dict().keys()}")
            
            params_dict = zip(self.diffusion.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.diffusion.model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            logger.error(f"Error setting parameters: {str(e)}")
            raise

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data.
        
        Args:
            parameters (List[np.ndarray]): Initial model parameters
            config (Dict[str, str]): Training configuration
            
        Returns:
            Tuple[List[np.ndarray], int, Dict]: Updated parameters, number of samples, and metrics
        """
        try:
            logger.info("Starting training round...")
            self.set_parameters(parameters)
            
            # Train using trainer
            metrics = self.trainer.train()
            logger.info(f"Training metrics: {metrics}")
            
            # Log metrics
            if metrics:
                for k, v in metrics.items():
                    wandb.log({f"{self.client_id}/train/{k}": v})
                    
            return self.get_parameters(config), len(self.dataset), metrics
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local data.
        
        Args:
            parameters (List[np.ndarray]): Model parameters to evaluate
            config (Dict[str, str]): Evaluation configuration
            
        Returns:
            Tuple[float, int, Dict]: Loss value, number of samples, and metrics
        """
        try:
            logger.info("Starting evaluation...")
            self.set_parameters(parameters)
            
            # Evaluate using trainer
            metrics = self.trainer.validate()
            logger.info(f"Evaluation metrics: {metrics}")
            
            # Log metrics
            if metrics:
                for k, v in metrics.items():
                    wandb.log({f"{self.client_id}/eval/{k}": v})
                    
            loss = metrics.get('loss', 0.0) if metrics else 0.0
            return loss, len(self.dataset), metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise

class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run = wandb.init(
            project="federated-binary-diffusion",
            entity="ilatyarshia",
            name="server",
            group="federated-training"
        )

    def aggregate_fit(self, server_round, results, failures):
        logger.info(f"Round {server_round}: {len(results)} results, {len(failures)} failures")
        if not results:
            wandb.log({
                "round": server_round,
                "failures": len(failures),
                "success": 0
            })
            return [], {}
        
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        wandb.log({
            "round": server_round,
            "failures": len(failures),
            "success": len(results),
            **metrics_aggregated
        })
        
        return parameters_aggregated, metrics_aggregated

def start_server():
    logger.info("Starting server...")
    try:
        strategy = CustomFedAvg(
            min_fit_clients=2,
            min_available_clients=2,
            fraction_fit=1.0,
            fraction_evaluate=1.0
        )
        
        fl.server.start_server(
            server_address="[::]:8080",
            config=fl.server.ServerConfig(num_rounds=10),
            strategy=strategy
        )
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise

def start_client(config_path: str, client_id: str):
    logger.info(f"Starting client {client_id}...")
    try:
        client = BinaryDiffusionClient(config_path, client_id)
        fl.client.start_numpy_client(
            server_address="[::]:8080",
            client=client
        )
    except Exception as e:
        logger.error(f"Client error: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Binary Diffusion')
    parser.add_argument('--mode', choices=['server', 'client'], required=True)
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--client-id', type=str, help='Client ID')
    parser.add_argument('--wandb-key', type=str, required=True, help='W&B API key')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'server':
            start_server()
        else:
            if not args.config or not args.client_id:
                raise ValueError("Config path and client ID required for client mode")
            start_client(args.config, args.client_id)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

# import flwr as fl
# import torch
# import wandb
# from binary_diffusion_tabular import (
#     FixedSizeBinaryTableDataset,
#     SimpleTableGenerator,
#     FixedSizeTableBinaryDiffusionTrainer
# )
# import yaml
# from collections import OrderedDict
# from typing import Dict, List, Tuple
# import numpy as np

# class BinaryDiffusionClient(fl.client.NumPyClient):
#     def __init__(self, config_path: str, client_id: str):
#         self.client_id = client_id
        
#         # Initialize W&B for client
#         self.run = wandb.init(
#             project="federated-binary-diffusion",
#             entity="ilatyarshia",
#             name=f"client-{client_id}",
#             group="federated-training"
#         )
        
#         with open(config_path, 'r') as f:
#             self.config = yaml.safe_load(f)
        
#         # Modify dataloader settings for stability
#         self.config['trainer']['batch_size'] = 32  # Reduce batch size
#         self.config['trainer']['dataloader_workers'] = 0  # Disable multiprocessing
        
#         # Initialize dataset and model
#         self.dataset = FixedSizeBinaryTableDataset.from_config(self.config['data'])
#         self.model = self._init_model()
        
#         # Initialize trainer with modified config
#         self.trainer = FixedSizeTableBinaryDiffusionTrainer.from_config(
#             config=self.config,
#             logger=self.run
#         )
#         self.trainer.diffusion = self.model

#     def fit(self, parameters, config):
#         try:
#             self.set_parameters(parameters)
            
#             # Train with error handling
#             metrics = self.trainer.train()
            
#             # Log metrics
#             if metrics:
#                 wandb.log({f"{self.client_id}/{k}": v for k, v in metrics.items()})
            
#             # Get updated parameters
#             updated_parameters = self.get_parameters(config)
            
#             return updated_parameters, len(self.dataset), metrics
            
#         except Exception as e:
#             wandb.log({
#                 f"{self.client_id}/error": str(e),
#                 f"{self.client_id}/error_type": type(e).__name__
#             })
#             raise

#     def _init_model(self):
#         model_params = {
#             'data_dim': self.dataset.row_size,
#             'dim': self.config['model']['dim'],
#             'n_res_blocks': self.config['model']['n_res_blocks'],
#             'out_dim': (
#                 self.dataset.row_size * 2
#                 if self.config['diffusion']['target'] == "two_way"
#                 else self.dataset.row_size
#             ),
#             'task': self.config['data']['task'],
#             'conditional': self.dataset.conditional,
#             'n_classes': 0 if self.config['data']['task'] == "regression" else self.dataset.n_classes,
#             'classifier_free_guidance': self.config['trainer']['classifier_free_guidance']
#         }
#         return SimpleTableGenerator(**model_params).to('cuda')

#     def get_parameters(self, config) -> List[np.ndarray]:
#         # Log model parameters to W&B
#         params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
#         for name, param in zip(self.model.state_dict().keys(), params):
#             wandb.log({f"{self.client_id}/params/{name}/mean": param.mean()})
#             wandb.log({f"{self.client_id}/params/{name}/std": param.std()})
#         return params

#     def set_parameters(self, parameters: List[np.ndarray]) -> None:
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
#         self.set_parameters(parameters)
        
#         # Train the model
#         metrics = self.trainer.train()
        
#         # Log metrics to W&B
#         if metrics:
#             wandb.log({f"{self.client_id}/{k}": v for k, v in metrics.items()})
            
#         return self.get_parameters(config), len(self.dataset), metrics

#     def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
#         self.set_parameters(parameters)
#         # Implement evaluation metrics specific to your case
#         loss = 0.0  # Replace with actual evaluation
#         metrics = {"loss": loss}
#         wandb.log({f"{self.client_id}/eval/{k}": v for k, v in metrics.items()})
#         return loss, len(self.dataset), metrics

# class CustomFedAvg(fl.server.strategy.FedAvg):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Initialize W&B for server
#         self.run = wandb.init(
#             project="federated-binary-diffusion",
#             entity="ilatyarshia",
#             name="server",
#             group="federated-training"
#         )

#     def aggregate_fit(self, *args, **kwargs):
#         results = super().aggregate_fit(*args, **kwargs)
#         if results is not None:
#             parameters, metrics = results
#             # Log aggregated metrics
#             wandb.log({"server/aggregated_metrics": metrics})
#             # Log parameter statistics
#             for i, param in enumerate(parameters):
#                 wandb.log({
#                     f"server/aggregated_params/layer_{i}/mean": param.mean(),
#                     f"server/aggregated_params/layer_{i}/std": param.std()
#                 })
#         return results

# def start_server():
#     strategy = CustomFedAvg(
#         min_fit_clients=2,
#         min_available_clients=2
#     )
    
#     fl.server.start_server(
#         server_address="[::]:8080",
#         config=fl.server.ServerConfig(num_rounds=10),
#         strategy=strategy
#     )

# def start_client(config_path: str, client_id: str):
#     client = BinaryDiffusionClient(config_path, client_id)
#     fl.client.start_numpy_client(
#         server_address="[::]:8080",
#         client=client
#     )

# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Federated Binary Diffusion with W&B logging')
#     parser.add_argument('--mode', choices=['server', 'client'], required=True)
#     parser.add_argument('--config', type=str, help='Path to config file')
#     parser.add_argument('--client-id', type=str, help='Client ID')
#     parser.add_argument('--wandb-key', type=str, required=True, help='W&B API key')
#     args = parser.parse_args()

#     # Initialize W&B
#     wandb.login(key=args.wandb_key)

#     if args.mode == 'server':
#         start_server()
#     else:
#         if not args.config or not args.client_id:
#             raise ValueError("Config path and client ID required for client mode")
#         start_client(args.config, args.client_id)