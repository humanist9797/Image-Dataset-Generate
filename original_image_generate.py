import cv2
import numpy as np
import os

# Set input and output directories
input_dir = 'D:/image/dataset1/original'  # Directory containing 45 images with various names
output_dir = 'D:/image/dataset1/original_output'  # Directory to save the resulting images
os.makedirs(output_dir, exist_ok=True)

# Set noise intensity range (standard deviation, randomly selected within min~max range)
min_noise_std = 10  # Minimum noise intensity
max_noise_std = 50  # Maximum noise intensity

# Get list of all image files in the input directory
image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

# Process each image
for file_name in image_files:
    # Image file path
    input_path = os.path.join(input_dir, file_name)
    
    # Load image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Image load failed: {input_path}")
        continue
    
    # Resize to 512x512 (using INTER_LINEAR interpolation)
    resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Generate random Gaussian noise intensity
    noise_std = np.random.uniform(min_noise_std, max_noise_std)
    
    # Generate random Gaussian noise (same size as grayscale image)
    noise = np.random.normal(0, noise_std, gray.shape).astype(np.float32)
    
    # Add noise (calculate in float and clip to prevent overflow)
    noisy = gray.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Get original file name without extension
    base_name = os.path.splitext(file_name)[0]
    
    # Output file path (tif extension with original base name)
    output_path = os.path.join(output_dir, f'{base_name}.tif')
    
    # Save image (in tif format)
    cv2.imwrite(output_path, noisy)
    
    print(f"Processing complete: {output_path} (Noise intensity: {noise_std:.2f})")