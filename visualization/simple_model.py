import torch
import torch.nn as nn

class SimpleConvModel(nn.Module):
    def __init__(self):
        super(SimpleConvModel, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=15, out_channels=1, kernel_size=1)
        
    def forward(self, x):
        # x has shape [5, 3, 128, 128]
        batch_size, channels, height, width = x.shape
        
        # Reshape to [1, 5*3, 128, 128] = [1, 15, 128, 128]
        x = x.reshape(1, batch_size * channels, height, width)
        
        # Apply convolution to get [1, 1, 128, 128]
        x = self.conv(x)
        
        # Reshape to [1, 128, 128]
        x = x.squeeze(1)
        
        return x
    
class LastImgModel(nn.Module):
    def __init__(self):
        super(LastImgModel, self).__init__()
        
        
    def forward(self, x):
        # x has shape [5, 3, 128, 128]
        # take the last image and average the 3 channels to get [1, 128, 128]

                # Select the last image in the batch
        x = x[-1]  # Shape [3, 128, 128]

        # Average over the channel dimension
        x = x.mean(dim=0, keepdim=True)  # Shape [1, 128, 128]

        return x


# Example usage
if __name__ == "__main__":
    # Create a random input tensor of shape [5, 3, 128, 128]
    input_tensor = torch.randn(5, 3, 128, 128)
    
    # Initialize the model
    model = SimpleConvModel()
    
    # Forward pass
    output = model(input_tensor)
    
    # Check output shape
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")

    # Print stats about conv layer weights
    print(f"Conv weights: min={model.conv.weight.min().item()}, max={model.conv.weight.max().item()}, mean={model.conv.weight.mean().item()}")
    print(f"Contains NaN: {torch.isnan(model.conv.weight).any().item()}")
