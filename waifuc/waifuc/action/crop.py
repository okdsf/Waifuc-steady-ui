from waifuc.action.base import ProcessAction
from waifuc.model import ImageItem
from PIL import Image
import os
from typing import Iterator, Optional
from imgutils.detect import detect_person, detect_heads, detect_halfbody, detect_eyes
from imgutils.segment import segment_rgba_with_isnetis
from ..model import ImageItem


class SmartCropAction(ProcessAction):
    def __init__(self, width=1024, height=1351):
        self.target_width = int(width)
        self.target_height = int(height)
        # print(f"SmartCropAction initialized. Target: {self.target_width}x{self.target_height}")

        self.MIN_AREA_FACTOR = 0.0005 
        self.MIN_DIMENSION = 10       
        self.HEAD_ROOM_FACTOR = 0.15 # 脸部上方头顶空间为脸部高度的百分比

    def _get_bbox_from_detection_list(self, detections):
        if detections and isinstance(detections, list) and len(detections) > 0:
            first_detection = detections[0]
            if isinstance(first_detection, (list, tuple)) and len(first_detection) > 0:
                box = first_detection[0]
                if isinstance(box, tuple) and len(box) == 4 and all(isinstance(coord, (int, float)) for coord in box):
                    x1, y1, x2, y2 = map(int, box)
                    if x1 < x2 and y1 < y2: return (x1, y1, x2, y2)
        return None

    def _adjust_bbox_to_original_coords(self, bbox, origin_x, origin_y):
        if not bbox: return None
        return (bbox[0] + origin_x, bbox[1] + origin_y,
                bbox[2] + origin_x, bbox[3] + origin_y)

    def _determine_char_box_and_features(self, original_image_rgb, img_width, img_height):
        """
        确定角色核心区 (char_box) 及其内部的关键特征 (脸部)。
        返回: (char_box_abs, char_box_img, face_bbox_in_char_img, primary_feature_global_bbox)
        char_box_abs: char_box在原图的坐标 (x1,y1,x2,y2)
        char_box_img: 从原图crop出的char_box图像
        face_bbox_in_char_img: 脸部在char_box_img内的坐标，可能为None
        primary_feature_global_bbox: 全局最优先的主体bbox (头/上半身/全身/IS_BBOX)
        """
        # print("DEBUG: _determine_char_box_and_features: START")
        best_subject_global_bbox = None # 在原图坐标系
        detection_priority = 0 
        initial_segmentation_bbox = None # IS_BBOX, 在原图坐标系
        
        search_area_img = original_image_rgb
        search_origin_abs = (0,0)

        # 1. IS_BBOX
        try:
            _, rgba_image = segment_rgba_with_isnetis(original_image_rgb)
            if rgba_image and hasattr(rgba_image, 'split') and len(rgba_image.split()) == 4:
                alpha = rgba_image.split()[3]
                is_bbox = alpha.getbbox()
                if is_bbox:
                    w, h = is_bbox[2]-is_bbox[0], is_bbox[3]-is_bbox[1]
                    if w >= self.MIN_DIMENSION and h >= self.MIN_DIMENSION and (w*h) >= (img_width*img_height*self.MIN_AREA_FACTOR):
                        initial_segmentation_bbox = is_bbox
                        # print(f"DEBUG: IS_BBOX found: {initial_segmentation_bbox}")
        except Exception: pass

        current_search_img = original_image_rgb
        current_search_origin_abs = (0,0)
        if initial_segmentation_bbox: # 如果IS_BBOX可靠，优先在IS_BBOX内搜索
            try:
                current_search_img = original_image_rgb.crop(initial_segmentation_bbox)
                current_search_origin_abs = (initial_segmentation_bbox[0], initial_segmentation_bbox[1])
            except: # crop失败，is_bbox可能无效
                initial_segmentation_bbox = None # 重置
                current_search_img = original_image_rgb
                current_search_origin_abs = (0,0)

        # 2. Head detection
        try:
            heads = detect_heads(current_search_img, **{})
            head_rel_bbox = self._get_bbox_from_detection_list(heads)
            if head_rel_bbox:
                best_subject_global_bbox = self._adjust_bbox_to_original_coords(head_rel_bbox, current_search_origin_abs[0], current_search_origin_abs[1])
                detection_priority = 4
        except Exception: pass

        # 3. HalfBody detection
        if detection_priority < 3:
            try:
                halfbodies = detect_halfbody(current_search_img, **{})
                hb_rel_bbox = self._get_bbox_from_detection_list(halfbodies)
                if hb_rel_bbox:
                    current_global_bbox = self._adjust_bbox_to_original_coords(hb_rel_bbox, current_search_origin_abs[0], current_search_origin_abs[1])
                    if 3 > detection_priority: best_subject_global_bbox = current_global_bbox; detection_priority = 3
            except Exception: pass
        
        # 4. Use IS_BBOX itself if it's the best so far
        if detection_priority < 2 and initial_segmentation_bbox:
             if 2 > detection_priority: best_subject_global_bbox = initial_segmentation_bbox; detection_priority = 2
        
        # 5. Person detection (on original full image, as fallback)
        if detection_priority < 1:
            try:
                persons = detect_person(original_image_rgb, **{}) # Always on full image
                person_abs_bbox = self._get_bbox_from_detection_list(persons)
                if person_abs_bbox:
                    if 1 > detection_priority: best_subject_global_bbox = person_abs_bbox; detection_priority = 1
            except Exception: pass

        # --- Define char_box based on findings ---
        char_box_abs = None
        if best_subject_global_bbox:
            # 如果IS_BBOX有效且包含best_subject_global_bbox，char_box优先基于IS_BBOX
            # (这里的“包含”判断可以基于IoU或者中心点是否在内，简化为中心点)
            bs_cx = (best_subject_global_bbox[0] + best_subject_global_bbox[2]) / 2
            bs_cy = (best_subject_global_bbox[1] + best_subject_global_bbox[3]) / 2
            if initial_segmentation_bbox and \
               initial_segmentation_bbox[0] <= bs_cx <= initial_segmentation_bbox[2] and \
               initial_segmentation_bbox[1] <= bs_cy <= initial_segmentation_bbox[3]:
                char_box_abs = initial_segmentation_bbox
                # print(f"DEBUG: char_box set to IS_BBOX {char_box_abs} as it contains/aligns with best_subject {best_subject_global_bbox}")
            else: # IS_BBOX无效或不包含，基于best_subject_global_bbox做极小扩展
                pad_w = (best_subject_global_bbox[2] - best_subject_global_bbox[0]) * 0.05 
                pad_h = (best_subject_global_bbox[3] - best_subject_global_bbox[1]) * 0.05
                char_box_abs = (
                    max(0, int(best_subject_global_bbox[0] - pad_w)),
                    max(0, int(best_subject_global_bbox[1] - pad_h)),
                    min(img_width, int(best_subject_global_bbox[2] + pad_w)),
                    min(img_height, int(best_subject_global_bbox[3] + pad_h))
                )
                # print(f"DEBUG: char_box derived from best_subject {best_subject_global_bbox} with min padding: {char_box_abs}")

        elif initial_segmentation_bbox: # 只有IS_BBOX
            char_box_abs = initial_segmentation_bbox
            # print(f"DEBUG: char_box set to IS_BBOX (no other subject found): {char_box_abs}")
        else: # 什么都没找到
            # print("DEBUG: No char_box could be determined.")
            return None, None, None, None

        if char_box_abs[0] >= char_box_abs[2] or char_box_abs[1] >= char_box_abs[3]: # 无效char_box
             # print(f"DEBUG: Invalid char_box dimensions: {char_box_abs}")
             return None, None, None, None
        
        try:
            char_box_img = original_image_rgb.crop(char_box_abs)
        except: # 防御性编程，crop可能因意外的char_box值失败
            # print(f"DEBUG: Failed to crop char_box_img with char_box_abs: {char_box_abs}")
            return None, None, None, None


        # 在char_box_img内部再次定位脸部，用于精细定位“取舍”的锚点
        face_bbox_in_char_img = None
        if char_box_img.width > 0 and char_box_img.height > 0 : #确保char_box_img有效
            try:
                heads_in_char_box = detect_heads(char_box_img, **{})
                face_bbox_in_char_img = self._get_bbox_from_detection_list(heads_in_char_box)
                # if face_bbox_in_char_img: print(f"DEBUG: Face found within char_box_img: {face_bbox_in_char_img}")
            except Exception: pass
        
        # print(f"DEBUG: _determine_char_box_and_features: char_box_abs={char_box_abs}, char_box_img_size={char_box_img.size if char_box_img else 'None'}, face_in_char_img={face_bbox_in_char_img}, primary_feature_global={best_subject_global_bbox}")
        return char_box_abs, char_box_img, face_bbox_in_char_img, best_subject_global_bbox


    def process(self, item):
        # print("\n--- SmartCropAction: process START (Strict No-Scale, User Logic V4 - Final Attempt) ---")
        if isinstance(item, ImageItem): original_image_input, meta = item.image, item.meta
        elif isinstance(item, Image.Image): original_image_input, meta = item, {}
        else: raise TypeError(f"预期输入 ImageItem 或 PIL.Image, 实际为 {type(item)}")

        if not isinstance(original_image_input, Image.Image):
             # print("ERROR: 输入对象不是有效的PIL Image")
             return ImageItem(Image.new('RGB', (self.target_width, self.target_height), (255,0,0)), meta) #红色错误图

        original_image_rgb = original_image_input
        if original_image_input.mode == 'RGBA':
            background = Image.new("RGB", original_image_input.size, (255,255,255))
            try:
                mask = original_image_input.split()[3]; background.paste(original_image_input, mask=mask)
                original_image_rgb = background
            except: original_image_rgb = original_image_input.convert('RGB')
        elif original_image_input.mode != 'RGB':
            original_image_rgb = original_image_input.convert('RGB')

        img_width, img_height = original_image_rgb.size
        # print(f"DEBUG: 原图 {img_width}x{img_height}, 目标 {self.target_width}x{self.target_height}")

        if img_width == 0 or img_height == 0:
            return ImageItem(Image.new('RGB', (self.target_width, self.target_height), (255,255,255)), meta)

        # --- 第一层逻辑分支：判断输出是否需要填充 ---
        # "直接裁剪"场景: 原图足够大，最终输出就是目标尺寸，无填充。
        requires_direct_crop_output = (img_width >= self.target_width and img_height >= self.target_height)
        # print(f"DEBUG: requires_direct_crop_output (无填充场景): {requires_direct_crop_output}")

        char_box_abs, char_box_img, face_bbox_in_char_img, primary_feature_global_bbox = \
            self._determine_char_box_and_features(original_image_rgb, img_width, img_height)

        final_cropped_content = None

        if not char_box_abs or not char_box_img: # 完全没有找到角色核心区
            # print("DEBUG: 未找到有效char_box。执行“盲裁”：从原图中央取内容。")
            # 这种情况下，一定会有取舍（如果原图大于目标）或填充（如果原图小于目标）
            # 抠图尺寸由min(原图尺寸, 目标尺寸)决定
            blind_crop_w = min(img_width, self.target_width)
            blind_crop_h = min(img_height, self.target_height)
            
            crop_x1 = (img_width - blind_crop_w) // 2
            crop_y1 = (img_height - blind_crop_h) // 2
            crop_x2 = crop_x1 + blind_crop_w
            crop_y2 = crop_y1 + blind_crop_h
            final_cropped_content = original_image_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        else:
            # print(f"DEBUG: 已确立 char_box_abs: {char_box_abs}, char_box_img尺寸: {char_box_img.size}")
            cb_img_w, cb_img_h = char_box_img.size

            # 确定从char_box中抠取内容的意图尺寸
            content_w_intended = min(cb_img_w, self.target_width)
            content_h_intended = min(cb_img_h, self.target_height)
            # print(f"DEBUG: 从char_box抠取意图尺寸: {content_w_intended}x{content_h_intended}")

            # 在char_box_img内部，根据“脸部优先，舍弃下方”原则定位这块内容
            crop_x1_in_cb, crop_y1_in_cb = 0, 0 # 默认从char_box_img的左上角开始取

            if face_bbox_in_char_img:
                # print(f"DEBUG: 使用脸部 {face_bbox_in_char_img} 在char_box内定位")
                fx1, fy1, fx2, fy2 = face_bbox_in_char_img
                fh = fy2 - fy1
                fcx = fx1 + (fx2 - fx1) / 2
                
                # 垂直定位：以脸部上方少量头顶空间开始，向下取content_h_intended
                headroom = fh * self.HEAD_ROOM_FACTOR
                crop_y1_in_cb = max(0, fy1 - headroom)
                crop_y2_in_cb = crop_y1_in_cb + content_h_intended
                
                if crop_y2_in_cb > cb_img_h: # 如果向下延伸超出了char_box_img的底部
                    crop_y2_in_cb = cb_img_h # 钳制到底部
                    crop_y1_in_cb = max(0, crop_y2_in_cb - content_h_intended) # 从底部向上回退，保证高度
                
                # 水平定位：以脸部中心为中心
                crop_x1_in_cb = fcx - content_w_intended / 2
                # (水平方向的钳制和调整如下)

            elif primary_feature_global_bbox: # 使用等级低一些的特征的中心
                # 将primary_feature_global_bbox转为char_box_img内的相对坐标
                pf_abs_x1, pf_abs_y1, pf_abs_x2, pf_abs_y2 = primary_feature_global_bbox
                pf_rel_x1 = pf_abs_x1 - char_box_abs[0]
                pf_rel_y1 = pf_abs_y1 - char_box_abs[1]
                pf_rel_x2 = pf_abs_x2 - char_box_abs[0]
                pf_rel_y2 = pf_abs_y2 - char_box_abs[1]
                
                pf_rel_cx = (pf_rel_x1 + pf_rel_x2) / 2
                pf_rel_cy = (pf_rel_y1 + pf_rel_y2) / 2

                crop_x1_in_cb = pf_rel_cx - content_w_intended / 2
                crop_y1_in_cb = pf_rel_cy - content_h_intended / 2 # 简单居中
            else: # 无内部特征，char_box本身就是主体，居中取
                crop_x1_in_cb = (cb_img_w - content_w_intended) / 2
                crop_y1_in_cb = (cb_img_h - content_h_intended) / 2

            # 确保选区不超出char_box_img边界 (精细钳制)
            final_crop_x1_in_cb = int(round(max(0, crop_x1_in_cb)))
            final_crop_y1_in_cb = int(round(max(0, crop_y1_in_cb)))
            # 确保 x2,y2 是基于期望尺寸和已钳制的x1,y1，然后再被char_box边界钳制
            final_crop_x2_in_cb = int(round(min(cb_img_w, final_crop_x1_in_cb + content_w_intended)))
            final_crop_y2_in_cb = int(round(min(cb_img_h, final_crop_y1_in_cb + content_h_intended)))
            # 再次修正x1,y1如果因为x2,y2的钳制导致尺寸不足
            final_crop_x1_in_cb = int(round(max(0, final_crop_x2_in_cb - content_w_intended)))
            final_crop_y1_in_cb = int(round(max(0, final_crop_y2_in_cb - content_h_intended)))

            # print(f"DEBUG: char_box内最终裁剪框: ({final_crop_x1_in_cb},{final_crop_y1_in_cb},{final_crop_x2_in_cb},{final_crop_y2_in_cb})")
            if final_crop_x1_in_cb >= final_crop_x2_in_cb or final_crop_y1_in_cb >= final_crop_y2_in_cb:
                # print("WARN: char_box内裁剪框无效，尝试使用整个char_box_img")
                final_cropped_content = char_box_img # 退回使用整个char_box_img
            else:
                final_cropped_content = char_box_img.crop((final_crop_x1_in_cb, final_crop_y1_in_cb, final_crop_x2_in_cb, final_crop_y2_in_cb))
        
        # print(f"DEBUG: final_cropped_content 尺寸: {final_cropped_content.size if final_cropped_content else 'None'}")

        # --- 第三阶段: 生成最终输出图像 ---
        if not final_cropped_content or final_cropped_content.width == 0 or final_cropped_content.height == 0:
            # print("ERROR: final_cropped_content 无效或为空，返回错误图")
            final_image = Image.new('RGB', (self.target_width, self.target_height), (255, 0, 0)) # 红色
            return ImageItem(final_image, meta)

        fcc_width, fcc_height = final_cropped_content.size

        if requires_direct_crop_output:

            if fcc_width == self.target_width and fcc_height == self.target_height:
                final_image = final_cropped_content
            else:
                # 这是不期望发生的情况：原图判定为够大，但抠出来的部分不够目标尺寸
                # 这表明char_box本身就小于目标尺寸，或者从char_box抠图的逻辑有问题
                # 按照“不缩放”原则，仍然粘贴到目标画布并填充
                # print(f"WARN: 期望直接裁剪输出，但抠出内容 ({fcc_width}x{fcc_height}) 与目标 ({self.target_width}x{self.target_height}) 不符。进行粘贴填充。")
                final_image = Image.new('RGB', (self.target_width, self.target_height), (255, 255, 255))
                paste_x = (self.target_width - fcc_width) // 2
                paste_y = (self.target_height - fcc_height) // 2
                final_image.paste(final_cropped_content, (paste_x, paste_y))
        else:
            # 原图至少有一边小于目标，或char_box抠出的内容小于目标，需要粘贴到画布并允许填充
            # print(f"DEBUG: 输出需要画布与填充: 抠出内容 ({fcc_width}x{fcc_height})，目标 ({self.target_width}x{self.target_height})")
            final_image = Image.new('RGB', (self.target_width, self.target_height), (255, 255, 255))
            paste_x = (self.target_width - fcc_width) // 2
            paste_y = (self.target_height - fcc_height) // 2
            final_image.paste(final_cropped_content, (paste_x, paste_y))
            
        # print(f"--- SmartCropAction: process END. Final image size: {final_image.size} ---")
        return ImageItem(final_image, meta)