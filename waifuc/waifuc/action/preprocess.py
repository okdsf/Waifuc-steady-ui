import logging
import math
import numpy as np
from PIL import Image
from typing import Tuple

from .base import ProcessAction
from ..model import ImageItem
from .esrgan import ESRGANAction

class PreprocessAction(ProcessAction):
    """
    一个通用的预处理动作，用于在最终裁剪前对图像进行筛选和尺寸归一化。
    它负责：
    1. 丢弃质量过低（需要放大倍数过高）的图像。
    2. 对过于冗余（有效信息远大于目标尺寸）的图像进行降采样。
    3. 对尺寸过小的图像进行超分辨率放大。
    4. 在所有缩放操作中，同步更新 'meta' 中的 'base_detection' 坐标。
    """

    def __init__(self, target_size: Tuple[int, int], 
                 downscale_threshold: float = 1.3,
                 upscale_discard_threshold: float = 5.0,
                 esrgan_model_path: str = 'esrgan-v0.2.3.pth'):
        self.target_w, self.target_h = target_size
        self.downscale_threshold = downscale_threshold
        self.upscale_discard_threshold = upscale_discard_threshold
        self.esrgan_model_path = esrgan_model_path

    def _update_meta_box(self, meta: dict, scale: float) -> dict:
        """按缩放比例更新meta中的包围盒信息"""
        if 'base_detection' in meta and 'box' in meta['base_detection']:
            new_meta = {**meta}
            x0, y0, x1, y1 = new_meta['base_detection']['box']
            new_meta['base_detection']['box'] = (
                x0 * scale,
                y0 * scale,
                x1 * scale,
                y1 * scale,
            )
            return new_meta
        return meta

    def process(self, item: ImageItem) -> ImageItem or None:
        if 'base_detection' not in item.meta or 'box' not in item.meta['base_detection']:
            logging.warning(f"跳过预处理，因为在 {item!r} 中未找到 'base_detection' 标注。")
            return item

        current_w, current_h = item.image.size
        bx0, by0, bx1, by1 = item.meta['base_detection']['box']
        base_w, base_h = bx1 - bx0, by1 - by0

        # 1. 质量筛选：检查放大系数是否过高
        # 注意: 放大系数基于当前图片尺寸，而不是原始检测框尺寸
        upscale_factor_w = self.target_w / current_w if current_w > 0 else float('inf')
        upscale_factor_h = self.target_h / current_h if current_h > 0 else float('inf')
        upscale_factor = max(upscale_factor_w, upscale_factor_h)

        if upscale_factor >= self.upscale_discard_threshold:
            logging.info(f"丢弃图像 {item!r}，因其所需放大系数 ({upscale_factor:.2f}) "
                         f"超过阈值 ({self.upscale_discard_threshold}).")
            return None # 丢弃此项目

        # 2. 冗余度检查：对过大图像进行降采样
        # 注意: 冗余度基于原始检测框尺寸
        oversize_ratio_w = base_w / self.target_w if self.target_w > 0 else float('inf')
        oversize_ratio_h = base_h / self.target_h if self.target_h > 0 else float('inf')
        oversize_ratio = max(oversize_ratio_w, oversize_ratio_h)

        if oversize_ratio > self.downscale_threshold:
            # 计算缩小比例，目标是将冗余度降到阈值以下
            # 我们将有效信息区域缩放到约等于目标尺寸
            downscale_to_ratio = self.target_w / base_w if base_w > base_h else self.target_h / base_h
            
            new_w = int(round(current_w * downscale_to_ratio))
            new_h = int(round(current_h * downscale_to_ratio))

            logging.info(f"图像 {item!r} 冗余度过高 ({oversize_ratio:.2f} > {self.downscale_threshold})，"
                         f"将其从 ({current_w}, {current_h}) 降采样至 ({new_w}, {new_h}).")
            
            new_image = item.image.resize((new_w, new_h), Image.LANCZOS)
            new_meta = self._update_meta_box(item.meta, downscale_to_ratio)
            return ImageItem(new_image, new_meta)

        # 3. 尺寸检查：对过小图像进行放大
        if upscale_factor > 1.0:
            # 向上取整到一位小数，以确保放大后尺寸足够
            scale = math.ceil(upscale_factor * 10) / 10
            logging.info(f"图像 {item!r} 尺寸过小，使用ESRGAN放大 {scale:.1f} 倍。")
            
            upscaler = ESRGANAction(scale=scale, model_path=self.esrgan_model_path)
            processed_item = upscaler.process(item) # ESRGANAction应该返回一个新的ImageItem
            new_meta = self._update_meta_box(item.meta, scale)
            return ImageItem(processed_item.image, new_meta)
            
        # 4. 如果尺寸合适，直接返回
        return item
