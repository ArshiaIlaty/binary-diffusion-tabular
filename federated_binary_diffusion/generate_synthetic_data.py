import torch
import pandas as pd
from pathlib import Path

from binary_diffusion_tabular import (
    BinaryDiffusion1D,
    SimpleTableGenerator,
    FixedSizeBinaryTableTransformation
)

def generate_synthetic_data(
    model_path: str,
    transformation_path: str,
    n_samples: int = 1000,
    batch_size: int = 64,
    device: str = "cuda"
):
    """Generate synthetic data using the trained global model"""
    
    # Load the trained global model
    ckpt = torch.load(model_path)
    
    # Initialize model and diffusion
    denoising_model = SimpleTableGenerator.from_config(ckpt['config_model']).to(device)
    diffusion = BinaryDiffusion1D.from_config(
        denoise_model=denoising_model,
        config=ckpt['config_diffusion']
    ).to(device)
    
    # Load model weights
    if 'diffusion_ema' in ckpt:  # Use EMA weights if available
        diffusion.load_ema(ckpt['diffusion_ema'])
    else:
        diffusion.load_state_dict(ckpt['diffusion'])
    
    # Load transformation
    transformation = FixedSizeBinaryTableTransformation.from_checkpoint(
        transformation_path
    )
    
    # Generate synthetic samples
    synthetic_samples = []
    remaining_samples = n_samples
    
    while remaining_samples > 0:
        current_batch_size = min(batch_size, remaining_samples)
        
        # Generate binary samples
        with torch.no_grad():
            x = diffusion.sample(
                n=current_batch_size,
                threshold=0.5,  # Threshold for binarization
                strategy="target"  # or "mask"
            )
        
        # Convert to tabular data
        x_df = transformation.inverse_transform(x)
        synthetic_samples.append(x_df)
        remaining_samples -= current_batch_size
        
        print(f"Generated {len(synthetic_samples) * batch_size} samples")
    
    # Combine all generated samples
    synthetic_df = pd.concat(synthetic_samples, ignore_index=True)
    return synthetic_df

if __name__ == "__main__":
    # Paths to your trained model and transformation
    model_path = "./models/global/global_model_final.pt"
    transformation_path = "./models/global/transformation.pt"
    output_path = "./results/synthetic_data.csv"
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(
        model_path=model_path,
        transformation_path=transformation_path,
        n_samples=1000  # Generate 1000 synthetic samples
    )
    
    # Save synthetic data
    synthetic_data.to_csv(output_path, index=False)
    print(f"Synthetic data saved to {output_path}")
    
    # Print some statistics
    print("\nOriginal vs Synthetic Data Statistics:")
    original_data = pd.read_csv("../data/cervical_train.csv")
    
    print("\nShape comparison:")
    print(f"Original data shape: {original_data.shape}")
    print(f"Synthetic data shape: {synthetic_data.shape}")
    
    print("\nColumn value ranges comparison:")
    for col in original_data.columns:
        print(f"\n{col}:")
        print(f"Original - min: {original_data[col].min()}, max: {original_data[col].max()}")
        print(f"Synthetic - min: {synthetic_data[col].min()}, max: {synthetic_data[col].max()}")
