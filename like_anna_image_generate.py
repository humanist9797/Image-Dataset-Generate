import cv2
import numpy as np
import os

# Set input image path and output directory
input_image_path = 'D:/image/dataset1/like_anna/like_anna.jpg'  # Path to your single input image (change as needed)
output_dir = 'D:/image/dataset1/like_anna_ouput'  # Directory to save the resulting images
os.makedirs(output_dir, exist_ok=True)

# Set noise intensity range (standard deviation, randomly selected within min~max range)
min_noise_std = 250 # Minimum noise intensity
max_noise_std = 500  # Maximum noise intensity

# Number of noisy images to generate
num_images = 1500

# Load the input image
img = cv2.imread(input_image_path)
if img is None:
    print(f"Image load failed: {input_image_path}")
    exit()

# Resize to 512x512 (using INTER_LINEAR interpolation)
resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

# Convert to grayscale
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Generate 1500 noisy images
for i in range(num_images):
    # Generate random Gaussian noise intensity
    noise_std = np.random.uniform(min_noise_std, max_noise_std)
    
    # Generate random Gaussian noise (same size as grayscale image)
    noise = np.random.normal(0, noise_std, gray.shape).astype(np.float32)
    
    # Add noise (calculate in float and clip to prevent overflow)
    noisy = gray.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Output file path (named 0000.tif to 1499.tif)
    output_path = os.path.join(output_dir, f'{i:04d}.tif')
    
    # Save image (in tif format)
    cv2.imwrite(output_path, noisy)
    
    print(f"Processing complete: {output_path} (Noise intensity: {noise_std:.2f})")