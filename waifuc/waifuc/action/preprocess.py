import logging
import math
import copy
from PIL import Image
from typing import Tuple, Optional, Dict, Any, List

from .base import ProcessAction
from ..model import ImageItem
from .esrgan import ESRGANAction

class PreprocessAction(ProcessAction):
    """
    一个通用的预处理动作，作为仿射变换的核心步骤。
    它负责在最终构图前，仅基于当前图像的尺寸进行智能的筛选和归一化。
    功能包括：
    1. 丢弃尺寸过小（需要放大倍数过高）的图像。
    2. 对尺寸远大于目标尺寸的图像进行降采样。
    3. 对尺寸小于目标尺寸的图像进行超分辨率放大。
    4. 在所有缩放操作中，通过更新 'affine_scale' 因子来精确记录几何变换。
    """
    def __init__(self, target_size: Tuple[int, int],
                 downscale_threshold: float = 1.3,
                 upscale_discard_threshold: float = 5.0,
                 esrgan: Optional[Dict[str, Any]] = None):
        """
        :param target_size: 最终构图的目标尺寸 (width, height)。
        :param downscale_threshold: 降采样阈值。当图像尺寸大于目标尺寸的这个倍数时，触发降采样。
        :param upscale_discard_threshold: 放大丢弃阈值。当图像需要放大超过这个倍数才能达到目标尺寸时，丢弃该图像。
        :param esrgan: ESRGANAction的配置字典。
        """
        self.target_w, self.target_h = target_size
        self.downscale_threshold = downscale_threshold
        self.upscale_discard_threshold = upscale_discard_threshold
        self.esrgan_config = esrgan or {}
        logging.basicConfig(level=logging.INFO)

    def process(self, item: ImageItem) -> Optional[ImageItem]:
        """
        处理单个ImageItem，根据其当前尺寸进行缩放或丢弃。
        """
        current_w, current_h = item.image.size
        if current_w <= 0 or current_h <= 0:
            logging.warning(f"跳过预处理 {item!r}，因为其图像尺寸无效 ({current_w}x{current_h}).")
            return item  # 对于无效尺寸的图像，跳过处理

        # --- 1. 质量筛选：基于当前图像尺寸判断放大需求 ---
        # 计算需要将当前图片放大多少倍才能达到目标尺寸
        required_upscale_w = self.target_w / current_w
        required_upscale_h = self.target_h / current_h
        required_upscale = max(required_upscale_w, required_upscale_h)

        # 如果需要的放大倍数超过了丢弃阈值，则返回None
        if required_upscale >= self.upscale_discard_threshold:
            logging.info(f"丢弃图像 {item!r}。图像需要放大 "
                         f"({required_upscale:.2f}x)，超过了阈值 ({self.upscale_discard_threshold}).")
            return None  # 丢弃此项目

        # --- 2. 冗余度检查：对过大图像进行降采样 ---
        # 计算当前图片尺寸是目标尺寸的多少倍
        oversize_ratio_w = current_w / self.target_w
        oversize_ratio_h = current_h / self.target_h
        oversize_ratio = max(oversize_ratio_w, oversize_ratio_h)

        # 如果尺寸超过了降采样阈值，则进行缩小
        if oversize_ratio > self.downscale_threshold:
            # 计算缩小比例，目标是将最大的维度缩小到约等于目标尺寸
            downscale_factor = 1.0 / oversize_ratio
            
            new_w = int(round(current_w * downscale_factor))
            new_h = int(round(current_h * downscale_factor))

            if new_w > 0 and new_h > 0:
                logging.info(f"缩小图像 {item!r} (尺寸超大 {oversize_ratio:.2f}x)，缩小因子为 "
                             f"{downscale_factor:.2f}，新尺寸为 ({new_w}, {new_h}).")
                
                new_image = item.image.resize((new_w, new_h), Image.LANCZOS)
                
                # 更新meta，记录下这次仿射变换
                new_meta = copy.deepcopy(item.meta)
                if 'geometric_info' not in new_meta:
                    new_meta['geometric_info'] = {}
                current_scale = new_meta['geometric_info'].get('affine_scale', 1.0)
                new_meta['geometric_info']['affine_scale'] = current_scale * downscale_factor
                return ImageItem(new_image, new_meta)

        # --- 3. 尺寸检查：对尺寸过小的图像进行放大 ---
        # 此逻辑在图像尺寸小于目标尺寸，但未达到丢弃阈值时触发
        if required_upscale > 1.0:
            # 使用ESRGAN进行放大
            # 向上取整到一位小数，以确保放大后尺寸足够
            upscale_s = math.ceil(required_upscale * 10) / 10
            
            logging.info(f"使用ESRGAN放大图像 {item!r}，放大倍数为 {upscale_s:.1f}x。")
            
            upscaler = ESRGANAction(scale=upscale_s, **self.esrgan_config)
            # 假设ESRGANAction只处理图片，我们需要自己包装ImageItem
            new_image = upscaler.process(item).image # 获取放大后的图片
            
            # 更新meta，记录下这次仿射变换
            new_meta = copy.deepcopy(item.meta)
            if 'geometric_info' not in new_meta:
                new_meta['geometric_info'] = {}
            current_scale = new_meta['geometric_info'].get('affine_scale', 1.0)
            new_meta['geometric_info']['affine_scale'] = current_scale * upscale_s
            return ImageItem(new_image, new_meta)
            
        # --- 4. 如果尺寸合适，无需任何操作，直接返回 ---
        logging.info(f"图像 {item!r} 尺寸合适，无需预处理。")
        return item