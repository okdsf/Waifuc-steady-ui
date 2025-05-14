import os
import numpy as np
import cv2
import torch
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from waifuc.action.base import ProcessAction
from waifuc.model import ImageItem
from torch.cuda.amp import autocast

class ESRGANAction(ProcessAction):
    def __init__(self, scale: float, model_path: str = None):
        """Initialize ESRGANAction with scaling factor and model path.

        Args:
            scale (float): Target scaling factor, e.g., 1.2, 2.0.
            model_path (str, optional): Path to Real-ESRGAN model file. Defaults to
                'C:\\Users\\Administrator\\Desktop\\AA\\Real-ESRGAN\\weights\\RealESRGAN_x4plus.pth'.
        """
        self.scale = scale
        self.model_path = model_path or r'C:\Users\Administrator\Desktop\AA\Real-ESRGAN\weights\RealESRGAN_x4plus.pth'
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create RRDBNet model (for RealESRGAN_x4plus)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # Initialize RealESRGANer with RRDBNet model and tile settings
        self.model = RealESRGANer(
            scale=4,  # Native model scale
            model_path=self.model_path,
            model=model,  # Explicitly provide RRDBNet model
            device='cuda' if torch.cuda.is_available() else 'cpu',
            tile=256,  
            tile_pad=64  # Use 64-pixel padding for tile overlap
        )

    def process(self, item: ImageItem) -> ImageItem:
        """Process an ImageItem by enhancing its image with Real-ESRGAN.

        Args:
            item (ImageItem): Input ImageItem containing the image and metadata.

        Returns:
            ImageItem: New ImageItem with enhanced image and original metadata.
        """
        # Extract PIL.Image
        image = item.image
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # PIL.Image (RGB) -> NumPy (RGB)
        np_img = np.array(image)
        
        # NumPy (RGB) -> NumPy (BGR)
        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        
        # Enhance image with Real-ESRGAN using FP16 mixed precision
        with torch.no_grad():
            with autocast():
                enhanced_bgr, _ = self.model.enhance(bgr_img, outscale=self.scale)
        
        # NumPy (BGR) -> NumPy (RGB)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        # NumPy (RGB) -> PIL.Image (RGB)
        enhanced_pil = Image.fromarray(enhanced_rgb)
        
        # Return new ImageItem with enhanced image and original metadata
        return ImageItem(enhanced_pil, item.meta)