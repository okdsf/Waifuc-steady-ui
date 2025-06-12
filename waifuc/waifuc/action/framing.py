import logging
from PIL import Image
import numpy as np
from typing import Tuple, Optional, Dict, Any, List

from .base import ProcessAction
from ..model import ImageItem

# 设置日志记录
logging.basicConfig(level=logging.INFO)

class FramingCropAction(ProcessAction):
    """
    一个利用上游传入的 'geometric_info' 进行智能构图裁剪的下游模块。
    它不再自己进行耗时的图像检测，而是直接利用元数据来定位和构图。
    默认使用轮廓质心进行构图，如果失败则回退到包围盒中心。
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

    def _get_contour_centroid(self, item: ImageItem) -> Optional[Tuple[float, float]]:
        """
        从元数据中计算轮廓的质心在当前图像坐标系下的位置。
        如果必要信息缺失，则返回 None。
        """
        try:
            geo_info = item.meta.get('geometric_info', {})
            relative_contours = geo_info.get('relative_contours')
            affine_scale = geo_info.get('affine_scale', 1.0)
            source_w, source_h = geo_info.get('source_image_size')
            crop_box = geo_info.get('crop_in_source')

            # 如果关键信息缺失，则无法进行精确计算
            if not relative_contours or not source_w or not source_h or not crop_box:
                return None

            # 将所有轮廓的所有点合并为一个 numpy 数组
            all_points_relative = np.array([point for contour in relative_contours for point in contour])
            if all_points_relative.size == 0:
                return None

            # 1. 将相对于原始大图的坐标，转换为在原始大图上的绝对像素坐标
            abs_points = all_points_relative * np.array([source_w, source_h])
            
            # 2. 减去切片在原始大图的偏移，得到在“未缩放切片”上的局部坐标
            crop_offset_x, crop_offset_y = crop_box[0], crop_box[1]
            local_points_prescale = abs_points - np.array([crop_offset_x, crop_offset_y])

            # 3. 乘以仿射缩放因子，得到在当前（可能已缩放）图像上的最终局部坐标
            final_local_points = local_points_prescale * affine_scale
            
            # 4. 计算最终质心
            centroid = np.mean(final_local_points, axis=0)
            return centroid[0], centroid[1]

        except (KeyError, IndexError, TypeError, ValueError) as e:
            logging.warning(f"计算轮廓质心时出错: {e}. 将回退到包围盒中心。")
            return None

    def _get_anchor_box_from_meta(self, item: ImageItem) -> Optional[Tuple[int, int, int, int]]:
        """
        从元数据中获取在当前图像坐标系下的绝对像素锚点边界框。
        """
        geo_info = item.meta.get('geometric_info', {})
        relative_features = geo_info.get('relative_features', {})
        affine_scale = geo_info.get('affine_scale', 1.0)
        
        current_w, current_h = item.image.size
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

        r_x1, r_y1, r_x2, r_y2 = relative_features[anchor_key]
        
        orig_abs_x1 = r_x1 * original_crop_size[0]
        orig_abs_y1 = r_y1 * original_crop_size[1]
        orig_abs_x2 = r_x2 * original_crop_size[0]
        orig_abs_y2 = r_y2 * original_crop_size[1]

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
            return self._simple_center_crop(item)

        current_image = item.image
        img_w, img_h = current_image.size

        # --- 1. 从元数据中获取锚点 ---
        anchor_box = self._get_anchor_box_from_meta(item)
        
        content_w = min(img_w, self.target_w)
        content_h = min(img_h, self.target_h)

        # --- 2. 根据锚点计算裁剪框 ---
        if anchor_box:
            ax1, ay1, ax2, ay2 = anchor_box
            anchor_w, anchor_h = ax2 - ax1, ay2 - ay1
            box_cx, box_cy = (ax1 + ax2) / 2, (ay1 + ay2) / 2

            # 默认使用包围盒中心作为构图中心
            comp_cx, comp_cy = box_cx, box_cy

            # 尝试使用轮廓质心作为更优的构图中心
            centroid = self._get_contour_centroid(item)
            if centroid:
                logging.info(f"使用轮廓质心 {centroid} 进行构图。")
                comp_cx, comp_cy = centroid
            else:
                logging.info("无法获取轮廓质心，回退到使用包围盒中心构图。")
            
            # 水平方向：始终以计算出的构图中心为中心
            crop_x1 = comp_cx - content_w / 2

            # 垂直方向：优先考虑头部构图的特殊逻辑
            if 'head' in item.meta.get('geometric_info', {}).get('relative_features', {}):
                # 为头部上方留出空间，此逻辑基于头部包围盒，不应被质心覆盖
                headroom = anchor_h * self.headroom_ratio
                crop_y1 = ay1 - headroom
            else:
                # 其他情况（半身、全身），使用计算出的构图中心（质心或包围盒中心）
                crop_y1 = comp_cy - content_h / 2
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
        
        final_meta = item.meta.copy()

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
        paste_y = (self.target_h - cropped_content.height) // 2
        final_image.paste(cropped_content, (paste_x, paste_y))
        
        return ImageItem(final_image, item.meta)
