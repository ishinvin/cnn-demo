import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load a sample image
img = Image.open('lena.png').convert('L')  # Convert to grayscale
img = img.resize((256, 256))  # Resize for easier processing

# Convert to PyTorch tensor
img_tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0).unsqueeze(0) / 255.0

# Define standard kernels
def get_kernel(kernel_type='identity'):
    if kernel_type == 'identity':
        kernel = torch.tensor([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]], dtype=torch.float32)
    elif kernel_type == 'edge':
        kernel = torch.tensor([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=torch.float32)
    elif kernel_type == 'sobel_x':
        kernel = torch.tensor([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=torch.float32)
    elif kernel_type == 'sobel_y':
        kernel = torch.tensor([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]], dtype=torch.float32)
    elif kernel_type == 'sharpen':
        kernel = torch.tensor([[ 0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]], dtype=torch.float32)
    elif kernel_type == 'blur':
        kernel = torch.ones(3, 3, dtype=torch.float32) / 9.0
    elif kernel_type == 'gaussian':
        kernel = torch.tensor([[1, 2, 1],
                              [2, 4, 2],
                              [1, 2, 1]], dtype=torch.float32) / 16.0
    else:
        raise ValueError("Unknown kernel type")
    
    return kernel.unsqueeze(0).unsqueeze(0)

# Define kernel types and their display names
kernel_types = [
    ('identity', 'Identity'),
    ('edge', 'Edge Detection'),
    ('sobel_x', 'Sobel X'),
    ('sobel_y', 'Sobel Y'),
    ('sharpen', 'Sharpen'),
    ('blur', 'Blur'),
    ('gaussian', 'Gaussian Blur')
]

# Create a figure with subplots
plt.figure(figsize=(15, 10))

# Plot original image
plt.subplot(3, 3, 1)
plt.imshow(img_tensor.squeeze(), cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Apply and plot each kernel
for i, (kernel_type, title) in enumerate(kernel_types, start=2):
    kernel = get_kernel(kernel_type)
    output = F.conv2d(img_tensor, kernel, padding=1)
    
    plt.subplot(3, 3, i)
    plt.imshow(output.squeeze().detach(), cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()