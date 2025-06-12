import os
import copy
from typing import Iterator, Optional, Any, Dict, Tuple, List

import numpy as np
from PIL import Image
from imgutils.detect import detect_person, detect_heads, detect_halfbody
from imgutils.segment import segment_rgba_with_isnetis
from skimage.measure import find_contours

from .base import BaseAction
from ..model import ImageItem

def _normalize_box(box: Tuple[float, float, float, float], source_size: Tuple[int, int]) -> Tuple[float, float, float, float]:
    source_w, source_h = source_size
    if source_w == 0 or source_h == 0: return 0.0, 0.0, 0.0, 0.0
    x1, y1, x2, y2 = box
    return x1 / source_w, y1 / source_h, x2 / source_w, y2 / source_h

def _normalize_contours(contours: List[np.ndarray], source_size: Tuple[int, int]) -> List[List[Tuple[float, float]]]:
    source_w, source_h = source_size
    if source_w == 0 or source_h == 0: return []
    normalized_contours = []
    for contour in contours:
        norm_contour = [(point[1] / source_w, point[0] / source_h) for point in contour]
        normalized_contours.append(norm_contour)
    return normalized_contours

def _offset_box(box: Tuple[float, float, float, float], offset: Tuple[int, int]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    off_x, off_y = offset
    return x1 + off_x, y1 + off_y, x2 + off_x, y2 + off_y

class ThreeStageSplitAction(BaseAction):
    def __init__(self, person_conf: Optional[dict] = None, halfbody_conf: Optional[dict] = None, head_conf: Optional[dict] = None, head_scale: float = 1.5, split_person: bool = True, extract_mask: bool = True, keep_origin_tags: bool = False, return_person: bool = True, return_halfbody: bool = True, return_head: bool = True):
        self.person_conf, self.halfbody_conf, self.head_conf = dict(person_conf or {}), dict(halfbody_conf or {}), dict(head_conf or {})
        self.head_scale, self.split_person, self.extract_mask = head_scale, split_person, extract_mask
        self.keep_origin_tags, self.return_person, self.return_halfbody, self.return_head = keep_origin_tags, return_person, return_halfbody, return_head

    def iter(self, item: ImageItem) -> Iterator[ImageItem]:
        source_image_rgb = item.image
        source_w, source_h = source_image_rgb.size
        filebody, ext = os.path.splitext(item.meta.get('filename', 'unknown'))
        base_meta = {key: value for key, value in item.meta.items() if key != 'tags' or self.keep_origin_tags}

        geometric_info_master: Dict[str, Any] = {'source_image_size': (source_w, source_h), 'relative_contours': None, 'relative_features': {}, 'affine_scale': 1.0}

        if self.extract_mask:
            try:
                _, rgba_image = segment_rgba_with_isnetis(source_image_rgb)
                if rgba_image and rgba_image.mode == 'RGBA':
                    alpha_mask = np.array(rgba_image.split()[3])
                    contours = find_contours(alpha_mask, 0.8)
                    geometric_info_master['relative_contours'] = _normalize_contours(contours, (source_w, source_h))
            except Exception: pass
        
        person_detections = detect_person(source_image_rgb, **self.person_conf) if self.split_person else [((0, 0, source_w, source_h), 'person', 1.0)]

        for i, (person_box, _, _) in enumerate(person_detections, start=1):
            px, py, px2, py2 = person_box
            person_image = source_image_rgb.crop(person_box)
            person_w, person_h = person_image.size
            head_detects, half_detects = detect_heads(person_image, **self.head_conf), detect_halfbody(person_image, **self.halfbody_conf)

            person_geo_info = copy.deepcopy(geometric_info_master)
            person_geo_info['relative_features']['person'] = _normalize_box(person_box, (source_w, source_h))
            if head_detects: person_geo_info['relative_features']['head'] = _normalize_box(_offset_box(head_detects[0][0], (px, py)), (source_w, source_h))
            if half_detects: person_geo_info['relative_features']['halfbody'] = _normalize_box(_offset_box(half_detects[0][0], (px, py)), (source_w, source_h))

            if self.return_person:
                person_meta = {**base_meta, 'branch_type': 'person', 'geometric_info': copy.deepcopy(person_geo_info)}
                person_meta['geometric_info']['crop_in_source'] = person_box
                person_meta['filename'] = f'{filebody}_person{i}{ext}'
                yield ImageItem(person_image, person_meta)

            if self.return_halfbody and half_detects:
                (hx1, hy1, hx2, hy2) = half_detects[0][0]
                halfbody_image = person_image.crop((hx1, hy1, hx2, hy2))
                halfbody_meta = {**base_meta, 'branch_type': 'halfbody', 'geometric_info': copy.deepcopy(person_geo_info)}
                halfbody_meta['geometric_info']['crop_in_source'] = _offset_box((hx1, hy1, hx2, hy2), (px, py))
                halfbody_meta['filename'] = f'{filebody}_person{i}_halfbody{ext}'
                yield ImageItem(halfbody_image, halfbody_meta)

            if self.return_head and head_detects:
                (hx0, hy0, hx1, hy1) = head_detects[0][0]
                cx, cy, w, h = (hx0 + hx1) / 2, (hy0 + hy1) / 2, hx1 - hx0, hy1 - hy0
                box_size = max(w, h) * self.head_scale
                
                crop_x0, crop_y0 = cx - box_size / 2, cy - box_size / 2
                crop_x1, crop_y1 = cx + box_size / 2, cy + box_size / 2
                
                # <<<--- 关键修复：添加严格的边界检查和坐标钳制，杜绝黑边 --- >>>
                final_crop_x0, final_crop_y0 = int(max(0, crop_x0)), int(max(0, crop_y0))
                final_crop_x1, final_crop_y1 = int(min(person_w, crop_x1)), int(min(person_h, crop_y1))

                if final_crop_x0 < final_crop_x1 and final_crop_y0 < final_crop_y1:
                    head_crop_box_rel = (final_crop_x0, final_crop_y0, final_crop_x1, final_crop_y1)
                    head_image = person_image.crop(head_crop_box_rel)
                    head_meta = {**base_meta, 'branch_type': 'head', 'geometric_info': copy.deepcopy(person_geo_info)}
                    head_meta['geometric_info']['crop_in_source'] = _offset_box(head_crop_box_rel, (px, py))
                    head_meta['filename'] = f'{filebody}_person{i}_head{ext}'
                    yield ImageItem(head_image, head_meta)

    def reset(self): pass


class PersonSplitAction(BaseAction):
    def __init__(self, keep_original: bool = False, level: str = 'm', version: str = 'v1.1',
                 conf_threshold: float = 0.3, iou_threshold: float = 0.5, keep_origin_tags: bool = False):
        self.keep_original = keep_original
        self.level = level
        self.version = version
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.keep_origin_tags = keep_origin_tags

    def iter(self, item: ImageItem) -> Iterator[ImageItem]:
        detection = detect_person(item.image, self.level, self.version,
                                  conf_threshold=self.conf_threshold, iou_threshold=self.iou_threshold)

        if 'filename' in item.meta:
            filename = item.meta['filename']
            filebody, ext = os.path.splitext(filename)
        else:
            filebody, ext = None, None

        if self.keep_original:
            yield item

        for i, (area, type_, score) in enumerate(detection):
            new_meta = {
                **item.meta,
                'crop': {'type': type_, 'score': score},
            }
            if 'tags' in new_meta and not self.keep_origin_tags:
                del new_meta['tags']
            if filebody is not None:
                new_meta['filename'] = f'{filebody}_person{i}{ext}'
            yield ImageItem(item.image.crop(area), new_meta)

    def reset(self):
        pass


