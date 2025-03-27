import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import time
from simple_model import SimpleConvModel

class PixelwiseLoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Pixel-by-pixel loss function for SimpleConvModel.
        
        Args:
            reduction (str): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(PixelwiseLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, predictions, targets, mask=None):
        """
        Calculate pixel-wise loss.
        
        Args:
            predictions: Tensor of shape [1, 128, 128]
            targets: Tensor of shape [1, 128, 128]
            mask: Optional boolean mask of valid pixels [1, 128, 128]
        
        Returns:
            loss: The calculated loss value
        """
        # Calculate squared error for each pixel
        pixel_errors = (predictions - targets) ** 2
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask is boolean
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask, dtype=torch.bool)
            elif mask.dtype != torch.bool:
                mask = mask.bool()
                
            # Apply mask and count valid pixels
            pixel_errors = pixel_errors[mask]
            num_valid_pixels = mask.sum().item()
            
            # Return appropriate reduction
            if self.reduction == 'mean':
                # Avoid division by zero
                return pixel_errors.sum() / max(num_valid_pixels, 1)
            elif self.reduction == 'sum':
                return pixel_errors.sum()
            else:  # 'none'
                return pixel_errors
        else:
            # No mask, apply reduction directly
            if self.reduction == 'mean':
                return pixel_errors.mean()
            elif self.reduction == 'sum':
                return pixel_errors.sum()
            else:  # 'none'
                return pixel_errors


def train_step_example(model, inputs, targets, optimizer, class_mask=None):
    """
    Example of a single training step using the pixel-wise loss.
    
    Args:
        model: SimpleConvModel instance
        inputs: Input tensor of shape [5, 3, 128, 128]
        targets: Target tensor of shape [1, 128, 128] 
        optimizer: PyTorch optimizer
        class_mask: Optional mask indicating valid pixels to consider
    """
    # Convert inputs to float if necessary
    inputs = inputs.float() if inputs.dtype != torch.float32 else inputs
    
    # Replace NaN values with 0.5 in inputs
    inputs = torch.nan_to_num(inputs, nan=0)
    targets = torch.nan_to_num(targets, nan=0)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    predictions = model(inputs)


    # Calculate loss
    criterion = PixelwiseLoss(reduction='mean')
    loss = criterion(predictions, targets, mask=class_mask)
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    return loss.item()


def train_one_epoch(model, dataloader, optimizer, device, epoch_num):
    """
    Train the model for one epoch.
    
    Args:
        model: The SimpleConvModel instance
        dataloader: DataLoader for the dataset
        optimizer: PyTorch optimizer
        device: Device to run training on ('cuda' or 'cpu')
        epoch_num: Current epoch number for logging
        
    Returns:
        average_loss: Average loss over the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    processed_batches = 0
    
    print(f"Starting epoch {epoch_num}...")
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        # Get inputs and targets
        inputs = batch["inputs"].to(device)
        targets = batch["targets"][:, 0].to(device)  # Get EVI channel
        
        # Valid mask: where target is not nan and in valid range
        valid_mask = ~torch.isnan(targets) & (targets >= -1) & (targets <= 1)
        
        # Process each sample in batch
        batch_loss = 0.0
        for j in range(inputs.size(0)):
            sample_inputs = inputs[j]  # [5, 3, 128, 128]
            sample_targets = targets[j:j+1]  # [1, 128, 128]
            sample_mask = valid_mask[j:j+1]  # [1, 128, 128]
            sample_mask = None
            
            
            # Train step
            loss = train_step_example(model, sample_inputs, sample_targets, optimizer, sample_mask)
            if np.isnan(loss):
                loss = 0
            batch_loss += loss
            
        # Average loss for the batch
        batch_loss /= inputs.size(0)
        running_loss += batch_loss
        processed_batches += 1
        
        # if i % 5 == 0:
        #     print(f"Epoch {epoch_num}, Batch {i}, Loss: {batch_loss:.6f}")
    
    # Calculate average loss for the epoch
    average_loss = running_loss / max(processed_batches, 1)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch_num} completed in {epoch_time:.2f}s. Average loss: {average_loss:.6f}")
    
    return average_loss


class DummyDataset(Dataset):
    """
    A dummy dataset that creates random data for testing the model.
    """
    def __init__(self, num_samples=20):
        """
        Initialize the dummy dataset.
        
        Args:
            num_samples: Number of samples to generate
        """
        self.num_samples = num_samples
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            A dictionary with inputs and targets
        """
        # Create random inputs of shape [5, 3, 128, 128]
        # Using a seed based on the index for reproducibility
        np.random.seed(idx)
        inputs = np.random.rand(5, 3, 128, 128).astype(np.float32)
        
        # Create random targets of shape [1, 128, 128]
        # Using a simple function of the inputs for predictability
        targets = np.mean(inputs, axis=(0, 1), keepdims=True)[0]  # Shape [1, 128, 128]
        
        # Add some noise to make it non-trivial
        targets += np.random.normal(0, 0.1, targets.shape).astype(np.float32)
        
        # Clip to a reasonable range
        targets = np.clip(targets, 0, 1)
        
        return {
            "inputs": torch.tensor(inputs, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32)
        }


def train_model(model, dataloader, optimizer, device, num_epochs=10):
    """
    Train the model for multiple epochs.
    
    Args:
        model: The SimpleConvModel instance
        dataloader: DataLoader for the dataset
        optimizer: PyTorch optimizer
        device: Device to run training on ('cuda' or 'cpu')
        num_epochs: Number of epochs to train for
        
    Returns:
        losses: List of average losses for each epoch
    """
    losses = []
    best_loss = float('inf')
    best_model_state = None
    
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        epoch_loss = train_one_epoch(model, dataloader, optimizer, device, epoch)
        losses.append(epoch_loss)
        
        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model with loss: {best_loss:.6f}")
        
        # Save checkpoint every 5 epochs
        if epoch % 5 == 0 or epoch == num_epochs:
            checkpoint_path = f'model_checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s. Best loss: {best_loss:.6f}")
    
    # Save the best model
    torch.save({
        'model_state_dict': best_model_state,
        'loss': best_loss,
    }, 'best_model.pth')
    print("Best model saved to best_model.pth")
    
    return losses


if __name__ == "__main__":
    NUM_EPOCHS = 100
    NUM_TO_FILTER = 1000
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Import required modules for dataset
        from dataset import SeqGreenEarthNetDataset, custom_collate_fn, ADDITIONAL_INFO_DICT
        
        # Define info list for dataset
        info_list = list(ADDITIONAL_INFO_DICT.keys())
        
        # Create dataset
        print("Loading SeqGreenEarthNetDataset...")
        ds_train = SeqGreenEarthNetDataset(
            folder="visualization/train/",
            input_channels=["red", "green", "blue"],
            target_channels=["evi", "class"],
            additional_info_list=info_list,
            time=True,
            use_mask=True,
        )
        
        # Filter dataset for valid samples
        filtered_indices = []
        to_filter = min(len(ds_train), NUM_TO_FILTER)
        
        print(f"Filtering dataset, original size is {len(ds_train)} (checking {to_filter} samples)...")
        for i in range(to_filter):
            try:
                batch = ds_train[i]
                seq = batch["inputs"]
                
                total_pixel_count = seq.shape[0] * seq.shape[1] * seq.shape[2] * seq.shape[3]
                nan_count = np.isnan(seq).sum() / total_pixel_count
                
                target = batch["targets"][0][0]
                target_nan_count = np.isnan(target).sum() / (target.shape[0] * target.shape[1])
                
                if nan_count < 0.3 and target_nan_count < 0.5:
                    filtered_indices.append(i)
            except:
                pass
        
        print(f"Filtered dataset: {len(filtered_indices)} valid samples out of {to_filter} checked")
        
        # Create a filtered dataset
        class FilteredDataset(Dataset):
            def __init__(self, original_dataset, indices):
                self.dataset = original_dataset
                self.indices = indices
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]
        
        # Use the filtered dataset
        filtered_dataset = FilteredDataset(ds_train, filtered_indices)
        
        # Create dataloader
        batch_size = 4
        dataloader = DataLoader(
            filtered_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Using 0 workers to avoid potential issues
            collate_fn=custom_collate_fn  # Use custom collate function from dataset module
        )
        
        # Initialize model and move to device
        model = SimpleConvModel().to(device)
        model = model.float()  # Ensure model is in float32
        
        # Initialize optimizer
        learning_rate = 0.01
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Train for X epochs
        losses = train_model(model, dataloader, optimizer, device, num_epochs=NUM_EPOCHS)
        
        # Print final loss summary
        print("\nTraining summary:")
        for epoch, loss in enumerate(losses, 1):
            print(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        # Save the final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses[-1],
        }, 'test_checkpoint.pth')
        print("Final model saved to test_checkpoint.pth")
        
        # Plot loss curve if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(losses) + 1), losses, marker='o')
            plt.title('Training Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.savefig('training_loss.png')
            print("Loss curve saved to training_loss.png")
        except ImportError:
            print("Matplotlib not available, skipping loss curve plot.")
    
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc() 