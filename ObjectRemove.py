import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# Check if a GPU is available; otherwise, default to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Mask R-CNN model with updated weights
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
model.eval()

# Function to load the image and convert it into a tensor
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
    return img, img_tensor

# Detect objects using Mask R-CNN model
def detect_objects(img_tensor):
    with torch.no_grad():
        prediction = model(img_tensor)
    return prediction[0]  # Return the first prediction (batch size is 1)

# Remove objects by masking them out and creating the inpainting mask
def remove_objects(image, masks, threshold=0.5):
    image_array = np.array(image)
    for mask in masks:
        binary_mask = mask >= threshold
        binary_mask = binary_mask.squeeze().cpu().numpy()  # Move the mask back to CPU and squeeze
        # Set masked region to 0 (black), representing removed objects
        image_array[binary_mask] = 0
    return image_array

# Project textures dynamically based on the viewer's perspective (projective texture mapping)
def projective_texture_mapping(image_array, masks, threshold=0.5):
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    inpaint_mask = np.zeros_like(gray_image)
    
    for mask in masks:
        binary_mask = mask >= threshold
        binary_mask = binary_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
        inpaint_mask = cv2.bitwise_or(inpaint_mask, binary_mask)
    
    # Apply projective texture mapping: Inpaint using the Telea or Navier-Stokes method
    projected_texture_image = cv2.inpaint(image_array, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    return projected_texture_image

# Function to render the results and visualize how the view-dependent rendering is optimized
def display_images(original_image, masked_image, projected_texture_image):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image (Before Object Removal)')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(masked_image)
    plt.title('Masked Image (Objects Removed)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(projected_texture_image)
    plt.title('Inpainted Image (Projective Texture Mapped)')
    plt.axis('off')
    
    plt.show()

# Main pipeline to execute the process of removing objects and dynamically adjusting textures
def object_removal_pipeline(image_path):
    # Load the image
    original_image, img_tensor = load_image(image_path)
    
    # Detect objects and obtain masks
    prediction = detect_objects(img_tensor)
    masks = prediction['masks']
    
    # Remove objects from the scene by applying object masks
    masked_image = remove_objects(original_image, masks)
    
    # Apply projective texture mapping to restore background realistically
    projected_texture_image = projective_texture_mapping(masked_image, masks)
    
    # Display the original, masked, and texture-projected images
    display_images(original_image, masked_image, projected_texture_image)

# Run the pipeline with an example image
image_path = 'D:\\Numpy\\Image_Project\\img1.jpeg'  # Replace with the path to your image
object_removal_pipeline(image_path)
