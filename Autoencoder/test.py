import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


from Autoencoder.Autoencoder import Autoencoder


class AnomalyDetectionAutoencoder(object):
    def __init__(self, model_path, device=None):
        """
        Initialize the class with the path to the pre-trained autoencoder model.
        Loads the model and sets the device (GPU/CPU).
        """
        self.model_path = model_path
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model()
        
        # Transformation applied to images before feeding into the model (if input is cv2 image)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),            # Convert cv2 image (numpy) to PIL image to use torchvision transforms
            transforms.Resize((2048, 2048)),    # Resize to model input size
            transforms.ToTensor()               # Convert image to tensor
        ])
    
    def _load_model(self):
        """
        Internal function to load the pre-trained model and move it to the correct device.
        """
        model = Autoencoder()
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)  # Move model to the selected device
        model.eval()           # Set model to evaluation mode
        return model
    
    def _preprocess_image(self, input_image):
        """
        Preprocess the input image (either a cv2 image or a tensor).
        If it's a cv2 image (numpy array), apply transformations; otherwise, ensure it is a tensor on the correct device.
        """
        if isinstance(input_image, np.ndarray):
            # If input is a cv2 image (numpy array), apply transformations
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = self.transform(input_image).unsqueeze(0)  # Apply transformation and add batch dimension
        elif isinstance(input_image, torch.Tensor):
            # If input is already a tensor, check if batch dimension exists
            if input_image.dim() == 3:  # If no batch dimension, add it
                image = input_image.unsqueeze(0)
            else:
                image = input_image
        else:
            raise TypeError("Unsupported input image type. Must be a cv2 image (numpy array) or a PyTorch Tensor.")
        
        return image.to(self.device)  # Move the image tensor to the same device as the model
    
    def calculate_mse(self, original, reconstructed):
        """
        Calculate Mean Squared Error (MSE) between original and reconstructed images.
        """
        return np.mean((original - reconstructed) ** 2)
    
    def reconstruct_image(self, input_image):
        """
        Perform image reconstruction using the autoencoder and calculate the reconstruction error.
        The input image can be a cv2 Image or a PyTorch tensor.
        
        Returns:
        - reconstructed_image: the reconstructed image as a cv2 image (numpy array)
        - reconstruction_error: the calculated MSE between the original and reconstructed image
        """
        # Preprocess the input image
        image_tensor = self._preprocess_image(input_image)
        
        # Forward pass through the autoencoder to get the reconstructed image
        with torch.no_grad():
            reconstructed_tensor = self.model(image_tensor)
        
        # Move the original and reconstructed images back to the CPU and convert to numpy arrays
        original_image = image_tensor[0].cpu().numpy()              # (C, H, W)
        reconstructed_image = reconstructed_tensor[0].cpu().numpy() # (C, H, W)
        
        # Calculate reconstruction error (MSE)
        reconstruction_error = self.calculate_mse(original_image, reconstructed_image)
        
        # Convert the reconstructed image tensor back to a cv2 image (numpy array)
        reconstructed_image_np = np.transpose(reconstructed_image, (1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
        reconstructed_image_np = (reconstructed_image_np * 255).astype(np.uint8)  # Rescale to [0, 255] for image
        
        # Convert RGB back to BGR for OpenCV compatibility
        reconstructed_image_bgr = cv2.cvtColor(reconstructed_image_np, cv2.COLOR_RGB2BGR)
        
        return reconstructed_image_bgr, reconstruction_error


# How to Use
"""
#############################################################
if __name__ == '__main__':
    model_path = "autoencoder_Final.pth"
    # Initialize the anomaly detection class with the model path and input image object
    anomaly_detector = AnomalyDetectionAutoencoder(model_path)
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################
    # Example: Using a cv2 Image object directly
    image_path = "test.jpg"
    image_obj = cv2.imread(image_path)  # Load an image as an OpenCV object (numpy array)
    ########################################################################################################
    ########################################################################################################
    ########################################################################################################

    # Perform image reconstruction and get the reconstruction error
    reconstructed_image, reconstruction_error = anomaly_detector.reconstruct_image(image_obj)

    # Display the results
    print(f"Reconstruction Error (MSE): {reconstruction_error}")

    # Save the reconstructed image
    cv2.imwrite("test-output/reconstructed_output.jpg", reconstructed_image)
"""