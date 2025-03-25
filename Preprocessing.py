import os
import cv2
import numpy as np

# Function to preprocess annotated images
def preprocess_images(input_folder, output_folder, target_size=(224, 224)):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # List all files in the input folder
    files = os.listdir(input_folder)
    image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]  # Adjust extensions as necessary

    for image_file in image_files:
        # Read annotated image
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)

        # Perform preprocessing (e.g., resize, normalize, etc.)
        img_resized = cv2.resize(img, target_size)  # Resize to target size

        # Example: Normalize pixel values to range [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0

        # Save preprocessed image
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, img_normalized * 255.0)  # Convert back to [0, 255] for saving as image

        print(f'Preprocessed image saved: {image_file}')

# Example usage:
input_folder = 'D:\\Annoted_images\\1'  # Replace with your input folder path containing annotated images
output_folder = 'D:\\Preprocessed_images\\1'  # Replace with your desired output folder path for preprocessed images
preprocess_images(input_folder, output_folder)
