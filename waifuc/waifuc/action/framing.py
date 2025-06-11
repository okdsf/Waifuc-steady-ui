import logging
from PIL import Image
from typing import Tuple, Optional

from .base import ProcessAction
from ..model import ImageItem

class FramingCropAction(ProcessAction):
    """
    An action for framing and composition, strictly following the robust logic
    from the original 'crop.py' (SmartCropAction).

    It uses pre-existing 'meta' data for high performance, avoiding redundant
    detections. This version is designed to be the definitive fix for all
    previously reported issues like black borders, content shifting, and
    improper cropping by faithfully replicating a proven algorithm.
    """
    def __init__(self, size: Tuple[int, int], head_room_factor: float = 0.15):
        """
        Initializes the framing action.

        :param size: The target size of the final output (width, height).
        :param head_room_factor: Composition parameter for 'head' type, defining top space.
        """
        self.target_width, self.target_height = size
        self.HEAD_ROOM_FACTOR = head_room_factor

    def _get_anchor_box(self, meta: dict) -> Tuple[Optional[str], Optional[Tuple[float, float, float, float]]]:
        """
        Determines the highest priority anchor bounding box and its type from the meta.
        Priority: head > halfbody > base_detection
        Returns: (anchor_type, anchor_bounding_box)
        """
        contained = meta.get('contained_features', {})
        if 'head' in contained and contained['head'].get('box'):
            return 'head', contained['head']['box']
        
        base_detection = meta.get('base_detection', {})
        base_type = base_detection.get('type', '')
        if 'head' in base_type and base_detection.get('box'):
            return 'head', base_detection['box']
            
        if 'halfbody' in contained and contained['halfbody'].get('box'):
            return 'halfbody', contained['halfbody']['box']
            
        return base_type, base_detection.get('box')

    def process(self, item: ImageItem) -> ImageItem:
        # Load image data into memory to release file lock, preventing PermissionError on Windows.
        item.image.load()
        meta = item.meta
        
        # --- Start of logic block faithfully replicated from crop.py ---
        original_image_input = item.image

        # Ensure image is in RGB format with a white background
        original_image_rgb = original_image_input
        if original_image_input.mode in ['RGBA', 'LA', 'P']:
            background = Image.new("RGB", original_image_input.size, (255,255,255))
            background.paste(original_image_input, mask=original_image_input.split()[-1])
            original_image_rgb = background
        elif original_image_input.mode != 'RGB':
            original_image_rgb = original_image_input.convert('RGB')

        img_width, img_height = original_image_rgb.size

        if img_width == 0 or img_height == 0:
            return ImageItem(Image.new('RGB', (self.target_width, self.target_height), (255,0,0)), meta)

        # --- Adaptation Point ---
        # Instead of running detection, we derive the necessary boxes from meta.
        # Here, the input image IS the character box.
        anchor_type, anchor_box = self._get_anchor_box(meta)
        
        # Simulate the output of `_determine_char_box_and_features` from crop.py
        # In this context, the pre-cropped item.image is our entire universe.
        char_box_img = original_image_rgb if anchor_box else None
        
        # The primary feature is our anchor, with coordinates relative to char_box_img (which is item.image)
        primary_feature_bbox = anchor_box
        # The face box is the anchor box if the anchor is a head.
        face_bbox_in_char_img = anchor_box if (anchor_box and 'head' in anchor_type) else None
        
        # --- End of Adaptation Point ---

        final_cropped_content = None

        if not char_box_img:
            # Fallback "blind crop" logic from crop.py
            blind_crop_w = min(img_width, self.target_width)
            blind_crop_h = min(img_height, self.target_height)
            
            crop_x1 = (img_width - blind_crop_w) // 2
            crop_y1 = (img_height - blind_crop_h) // 2
            final_cropped_content = original_image_rgb.crop((crop_x1, crop_y1, crop_x1 + blind_crop_w, crop_y1 + blind_crop_h))
        else:
            cb_img_w, cb_img_h = char_box_img.size

            content_w_intended = min(cb_img_w, self.target_width)
            content_h_intended = min(cb_img_h, self.target_height)
            
            # --- Exactly the same positioning logic as in crop.py ---
            if face_bbox_in_char_img:
                fx1, fy1, fx2, fy2 = face_bbox_in_char_img
                fh = fy2 - fy1
                fcx = fx1 + (fx2 - fx1) / 2
                headroom = fh * self.HEAD_ROOM_FACTOR
                crop_y1_in_cb = max(0, fy1 - headroom)
                crop_x1_in_cb = fcx - content_w_intended / 2
            elif primary_feature_bbox:
                pf_x1, pf_y1, pf_x2, pf_y2 = primary_feature_bbox
                pf_cx = (pf_x1 + pf_x2) / 2
                pf_cy = (pf_y1 + pf_y2) / 2
                crop_x1_in_cb = pf_cx - content_w_intended / 2
                crop_y1_in_cb = pf_cy - content_h_intended / 2
            else: # Should not happen if anchor_box exists, but as a safeguard
                crop_x1_in_cb = (cb_img_w - content_w_intended) / 2
                crop_y1_in_cb = (cb_img_h - content_h_intended) / 2

            # --- Exactly the same robust clamping logic from crop.py ---
            # This ensures the crop box stays within the bounds of char_box_img
            final_crop_x1 = int(round(max(0, crop_x1_in_cb)))
            final_crop_y1 = int(round(max(0, crop_y1_in_cb)))

            final_crop_x2 = int(round(min(cb_img_w, final_crop_x1 + content_w_intended)))
            final_crop_y2 = int(round(min(cb_img_h, final_crop_y1 + content_h_intended)))

            # Recalculate top-left based on clamped bottom-right to preserve size
            final_crop_x1 = int(round(max(0, final_crop_x2 - content_w_intended)))
            final_crop_y1 = int(round(max(0, final_crop_y2 - content_h_intended)))

            if final_crop_x1 >= final_crop_x2 or final_crop_y1 >= final_crop_y2:
                final_cropped_content = char_box_img # Fallback to using the whole image
            else:
                final_cropped_content = char_box_img.crop((final_crop_x1, final_crop_y1, final_crop_x2, final_crop_y2))
        
        # --- Exactly the same final pasting logic from crop.py ---
        if not final_cropped_content or final_cropped_content.width == 0 or final_cropped_content.height == 0:
            final_image = Image.new('RGB', (self.target_width, self.target_height), (255, 0, 0)) # Red error image
            return ImageItem(final_image, meta)

        fcc_width, fcc_height = final_cropped_content.size
        
        final_image = Image.new('RGB', (self.target_width, self.target_height), (255, 255, 255))
        paste_x = (self.target_width - fcc_width) // 2
        paste_y = (self.target_height - fcc_height) // 2
        final_image.paste(final_cropped_content, (paste_x, paste_y))
            
        return ImageItem(final_image, meta)