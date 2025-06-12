import logging
from PIL import Image
from typing import Tuple, Optional, Dict, Any

from .base import ProcessAction
from ..model import ImageItem

# 设置日志记录
logging.basicConfig(level=logging.INFO)

class FramingCropAction(ProcessAction):
    """
    一个利用上游传入的 'geometric_info' 进行智能构图裁剪的下游模块。
    它不再自己进行耗时的图像检测，而是直接利用元数据来定位和构图。
    """
    def __init__(self, size: Tuple[int, int], headroom_ratio: float = 0.15):
        """
        :param size: 最终输出的目标尺寸 (width, height)。
        :param headroom_ratio: 基于头部高度，在其上方留出的头顶空间比例。
        """
        if not isinstance(size, (tuple, list)) or len(size) != 2:
            raise ValueError("Parameter 'size' must be a tuple of (width, height).")
        self.target_w, self.target_h = size
        self.headroom_ratio = headroom_ratio

    def _get_anchor_box_from_meta(self, item: ImageItem) -> Optional[Tuple[int, int, int, int]]:
        """
        从元数据中获取在当前图像坐标系下的绝对像素锚点边界框。
        """
        geo_info = item.meta.get('geometric_info', {})
        relative_features = geo_info.get('relative_features', {})
        affine_scale = geo_info.get('affine_scale', 1.0)
        
        # 这里的 source_size 是指分割出的那张初始图的尺寸，不是最最原始的大图
        # 但 PreprocessAction 并没有传递这个，它只传递了最原始的 source_image_size
        # 幸运的是，我们可以通过当前图片尺寸和缩放比例反推回去
        current_w, current_h = item.image.size
        # 初始尺寸 ≈ 当前尺寸 / 缩放比例
        original_crop_w = current_w / affine_scale
        original_crop_h = current_h / affine_scale
        original_crop_size = (original_crop_w, original_crop_h)

        anchor_key = None
        if 'head' in relative_features:
            anchor_key = 'head'
        elif 'halfbody' in relative_features:
            anchor_key = 'halfbody'
        elif 'person' in relative_features:
            anchor_key = 'person'
        
        if not anchor_key:
            return None

        # 获取相对坐标
        r_x1, r_y1, r_x2, r_y2 = relative_features[anchor_key]
        
        # 换算到初始分割图的绝对像素坐标
        orig_abs_x1 = r_x1 * original_crop_size[0]
        orig_abs_y1 = r_y1 * original_crop_size[1]
        orig_abs_x2 = r_x2 * original_crop_size[0]
        orig_abs_y2 = r_y2 * original_crop_size[1]

        # 再根据仿射变换比例，计算出在当前这张图上的绝对像素坐标
        current_abs_x1 = int(round(orig_abs_x1 * affine_scale))
        current_abs_y1 = int(round(orig_abs_y1 * affine_scale))
        current_abs_x2 = int(round(orig_abs_x2 * affine_scale))
        current_abs_y2 = int(round(orig_abs_y2 * affine_scale))

        return (current_abs_x1, current_abs_y1, current_abs_x2, current_abs_y2)

    def process(self, item: ImageItem) -> ImageItem:
        """
        处理单个图像项。
        """
        if 'geometric_info' not in item.meta:
            logging.warning(f"Skipping framing for {item!r} due to missing 'geometric_info'. Performing simple center crop.")
            # 降级方案：执行简单的中心裁剪
            return self._simple_center_crop(item)

        current_image = item.image
        img_w, img_h = current_image.size

        # --- 1. 从元数据中获取锚点 ---
        anchor_box = self._get_anchor_box_from_meta(item)
        
        # 确定从当前图像中抠取内容的尺寸
        content_w = min(img_w, self.target_w)
        content_h = min(img_h, self.target_h)

        # --- 2. 根据锚点计算裁剪框 ---
        if anchor_box:
            ax1, ay1, ax2, ay2 = anchor_box
            anchor_w, anchor_h = ax2 - ax1, ay2 - ay1
            anchor_cx, anchor_cy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
            
            # 水平方向：始终以锚点为中心
            crop_x1 = anchor_cx - content_w / 2

            # 垂直方向：优先考虑头部构图
            if 'head' in item.meta.get('geometric_info', {}).get('relative_features', {}):
                # 为头部上方留出空间
                headroom = anchor_h * self.headroom_ratio
                crop_y1 = ay1 - headroom - (content_h / 2 - (anchor_h / 2 + headroom)) # 复杂计算，确保锚点头部在裁剪框中合适位置
                crop_y1 = ay1 - headroom # 简化版本：从头顶上方预留空间处开始裁剪
            else:
                # 其他情况（半身、全身），简单居中
                crop_y1 = anchor_cy - content_h / 2
        else:
            # --- 3. 降级：无有效锚点，执行中心裁剪 ---
            logging.warning(f"Could not determine anchor from meta for {item!r}. Performing simple center crop.")
            return self._simple_center_crop(item)

        # --- 4. 钳制裁剪框，确保不越界 ---
        final_x1 = max(0, min(int(round(crop_x1)), img_w - content_w))
        final_y1 = max(0, min(int(round(crop_y1)), img_h - content_h))
        final_x2 = final_x1 + content_w
        final_y2 = final_y1 + content_h

        # 执行裁剪
        cropped_content = current_image.crop((final_x1, final_y1, final_x2, final_y2))

        # --- 5. 粘贴到目标画布 ---
        final_image = Image.new('RGB', (self.target_w, self.target_h), (255, 255, 255))
        paste_x = (self.target_w - cropped_content.width) // 2
        paste_y = (self.target_h - cropped_content.height) // 2
        final_image.paste(cropped_content, (paste_x, paste_y))
        
        # 在这里可以清空或更新meta中的几何信息，因为它已经被消费掉了
        final_meta = item.meta.copy()
        # final_meta.pop('geometric_info', None) # 可选

        return ImageItem(final_image, final_meta)

    def _simple_center_crop(self, item: ImageItem) -> ImageItem:
        """一个简单的中心裁剪降级方法"""
        current_image = item.image
        img_w, img_h = current_image.size
        content_w = min(img_w, self.target_w)
        content_h = min(img_h, self.target_h)
        
        crop_x1 = (img_w - content_w) // 2
        crop_y1 = (img_h - content_h) // 2
        crop_x2 = crop_x1 + content_w
        crop_y2 = crop_y1 + content_h
        
        cropped_content = current_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        
        final_image = Image.new('RGB', (self.target_w, self.target_h), (255, 255, 255))
        paste_x = (self.target_w - cropped_content.width) // 2
        paste_y = (self.target_height - cropped_content.height) // 2
        final_image.paste(cropped_content, (paste_x, paste_y))
        
        return ImageItem(final_image, item.meta)

