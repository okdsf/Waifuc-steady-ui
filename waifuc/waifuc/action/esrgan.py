import os
import numpy as np
import cv2
import torch
from PIL import Image
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet # 假设此导入路径是正确的
from waifuc.action.base import ProcessAction
from waifuc.model import ImageItem
from torch.cuda.amp import autocast

class ESRGANAction(ProcessAction):
    def __init__(self, scale: float, model_path: str = None):
        """Initialize ESRGANAction with scaling factor and model path.

        Args:
            scale (float): Target scaling factor, e.g., 1.2, 2.0.
            model_path (str, optional): Path or name of Real-ESRGAN model file.
                If None or a simple name, it might be resolved against a default directory.
        """
        self.scale = scale
        
        # 定义默认的模型权重目录和默认模型名称
        default_model_dir = r'C:\Users\Administrator\Desktop\AA\Real-ESRGAN\weights'
        default_model_name = 'RealESRGAN_x4plus.pth'
        
        resolved_model_path = model_path

        if resolved_model_path is None:
            #情况1：未提供 model_path，使用硬编码的默认完整路径
            resolved_model_path = os.path.join(default_model_dir, default_model_name)
        elif not os.path.isabs(resolved_model_path) and not os.path.exists(resolved_model_path):
            # 情况2：model_path 是一个相对路径（很可能只是文件名）并且它当前不存在。
            # 假定它是位于 default_model_dir 中的文件名。
            print(f"Model path '{resolved_model_path}' is relative and not found in current directory. Assuming it's in default directory: {default_model_dir}")
            resolved_model_path = os.path.join(default_model_dir, os.path.basename(resolved_model_path)) # 使用 os.path.basename 以防传入的是相对路径如 "subdir/model.pth"
        # 情况3：model_path 是一个绝对路径，或者是一个已经存在的相对路径。直接使用。
        # 此时 resolved_model_path 无需更改。

        self.model_path = resolved_model_path
        
        # 检查模型文件是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # 从模型文件名推断 num_block
        model_filename = os.path.basename(self.model_path)
        num_block = 23 # 默认值

        if 'RealESRGAN_x4plus_anime_6B' in model_filename: # 专门为 RealESRGAN_x4plus_anime_6B.pth
            num_block = 6
            print(f"Inferred num_block={num_block} for model: {model_filename}")
        elif 'RealESRGAN_x4plus' in model_filename: # RealESRGAN_x4plus.pth 的默认值
            num_block = 23
            print(f"Inferred num_block={num_block} for model: {model_filename}")
        else:
            # 如果模型名称不包含明确的线索，则使用默认值并发出警告
            print(f"Warning: Could not reliably determine num_block for model '{model_filename}'. Defaulting to {num_block}. "
                  "If this model has a different architecture, it may not load correctly.")

        # 使用推断出的 num_block 创建 RRDBNet 模型
        # 假设 RRDBNet 的其他参数 (num_in_ch, num_out_ch, num_feat, num_grow_ch, scale) 对于这些模型是通用的
        model_arch_scale = 4 # 大多数 RealESRGAN 模型是 x4 的
        rrdb_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=num_block, num_grow_ch=32, scale=model_arch_scale)
        
        # 使用 RRDBNet 模型和正确的模型路径初始化 RealESRGANer
        self.model = RealESRGANer(
            scale=model_arch_scale, # 这是模型本身的放大倍数
            model_path=self.model_path, # 这是权重文件的路径
            model=rrdb_model,         # 这是实例化的模型结构
            device='cuda' if torch.cuda.is_available() else 'cpu',
            tile=256,  
            tile_pad=64
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
            with autocast(): # autocast 用于混合精度，有助于提高效率
                # self.scale 是用户期望的输出放大倍数，可以与模型本身的放大倍数不同
                enhanced_bgr, _ = self.model.enhance(bgr_img, outscale=self.scale) 
        
        # NumPy (BGR) -> NumPy (RGB)
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        
        # NumPy (RGB) -> PIL.Image (RGB)
        enhanced_pil = Image.fromarray(enhanced_rgb)
        
        # Return new ImageItem with enhanced image and original metadata
        return ImageItem(enhanced_pil, item.meta)