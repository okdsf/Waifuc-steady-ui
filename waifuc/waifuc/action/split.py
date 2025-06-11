import os
from typing import Iterator, Optional

from imgutils.detect import detect_person, detect_heads, detect_halfbody, detect_eyes

from .base import BaseAction
from ..model import ImageItem


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


class ThreeStageSplitAction(BaseAction):
    
    def __init__(self, person_conf: Optional[dict] = None, halfbody_conf: Optional[dict] = None,
                 head_conf: Optional[dict] = None, head_scale: float = 1.5,
                 split_person: bool = True, keep_origin_tags: bool = False,
                 return_person: bool = True, return_halfbody: bool = True,
                 return_head: bool = True):
        self.person_conf = dict(person_conf or {})
        self.halfbody_conf = dict(halfbody_conf or {})
        self.head_conf = dict(head_conf or {})
        self.head_scale = head_scale
        self.split_person = split_person
        self.keep_origin_tags = keep_origin_tags
        self.return_person = return_person
        self.return_halfbody = return_halfbody
        self.return_head = return_head

    def iter(self, item: ImageItem) -> Iterator[ImageItem]:
        if 'filename' in item.meta:
            filename = item.meta['filename']
            filebody, ext = os.path.splitext(filename)
        else:
            filebody, ext = None, None

        person_detections = detect_person(item.image, **self.person_conf) if self.split_person else \
            [((0, 0, item.image.width, item.image.height), 'person', 1.0)]

        for i, (px, person_type, person_score) in enumerate(person_detections, start=1):
            person_image = item.image.crop(px)
            
            # --- 在 person 内部检测 head 和 halfbody ---
            head_detects = detect_heads(person_image, **self.head_conf)
            half_detects = detect_halfbody(person_image, **self.halfbody_conf)

            # --- 构造包含所有层级信息的元数据 ---
            contained_features = {}
            if head_detects:
                contained_features['head'] = {'box': head_detects[0][0], 'score': head_detects[0][2]}
            if half_detects:
                contained_features['halfbody'] = {'box': half_detects[0][0], 'score': half_detects[0][2]}
            
            base_meta = {**item.meta}
            if 'tags' in base_meta and not self.keep_origin_tags:
                del base_meta['tags']

            # --- 产出 Person Item ---
            if self.return_person:
                person_meta = {
                    **base_meta,
                    'base_detection': {'type': 'person', 'box': (0, 0, person_image.width, person_image.height)},
                    'contained_features': contained_features # 将内部特征一并传递
                }
                if filebody: person_meta['filename'] = f'{filebody}_person{i}{ext}'
                yield ImageItem(person_image, person_meta)

            # --- 产出 Half-body Item ---
            if self.return_halfbody and half_detects:
                (hx1, hy1, hx2, hy2), halfbody_type, halfbody_score = half_detects[0]
                halfbody_image = person_image.crop((hx1, hy1, hx2, hy2))
                # 对于 halfbody 来说，它自身就是 base_detection
                # 但我们也需要知道它内部是否包含 head
                halfbody_contained = {}
                if head_detects:
                    # 需要计算 head 在 halfbody 裁剪框内的相对坐标
                    (head_x0, head_y0, head_x1, head_y1) = head_detects[0][0]
                    if not (head_x1 < hx1 or head_x0 > hx2 or head_y1 < hy1 or head_y0 > hy2): # 确保head在halfbody内
                       halfbody_contained['head'] = {
                           'box': (head_x0 - hx1, head_y0 - hy1, head_x1 - hx1, head_y1 - hy1),
                           'score': head_detects[0][2]
                       }
                
                halfbody_meta = {
                    **base_meta,
                    'base_detection': {'type': halfbody_type, 'box': (0, 0, halfbody_image.width, halfbody_image.height)},
                    'contained_features': halfbody_contained,
                }
                if filebody: halfbody_meta['filename'] = f'{filebody}_person{i}_halfbody{ext}'
                yield ImageItem(halfbody_image, halfbody_meta)

            # --- 产出 Head Item ---
            if self.return_head and head_detects:
                (hx0, hy0, hx1, hy1), head_type, head_score = head_detects[0]
                cx, cy = (hx0 + hx1) / 2, (hy0 + hy1) / 2
                w, h = hx1 - hx0, hy1 - hy0
                box_size = max(w, h) * self.head_scale
                crop_x0, crop_y0 = int(max(cx - box_size/2, 0)), int(max(cy - box_size/2, 0))
                
                head_image = person_image.crop((crop_x0, crop_y0, crop_x0 + box_size, crop_y0 + box_size))
                
                head_meta = {
                    **base_meta,
                    'base_detection': { # 对于 head item, 其 base_detection 就是 head 本身
                        'type': head_type,
                        'box': (hx0 - crop_x0, hy0 - crop_y0, hx1 - crop_x0, hy1 - crop_y0)
                    },
                    'contained_features': {} # head 内部不再包含其他特征
                }
                if filebody: head_meta['filename'] = f'{filebody}_person{i}_head{ext}'
                yield ImageItem(head_image, head_meta)

                if self.split_eyes:
                    eye_detects = detect_eyes(head_image, **self.eye_conf)
                    for j, ((ex0, ey0, ex1, ey1), eye_type, eye_score) in enumerate(eye_detects):
                        cx, cy = (ex0 + ex1) / 2, (ey0 + ey1) / 2
                        width, height = ex1 - ex0, ey1 - ey0
                        width = height = max(width, height) * self.eye_scale
                        x0, y0 = int(max(cx - width / 2, 0)), int(max(cy - height / 2, 0))
                        x1, y1 = int(min(cx + width / 2, head_image.width)), \
                            int(min(cy + height / 2, head_image.height))
                        eye_image = head_image.crop((x0, y0, x1, y1))
                        eye_meta = {
                            **item.meta,
                            'crop': {'type': eye_type, 'score': eye_score},
                        }
                        if 'tags' in eye_meta and not self.keep_origin_tags:
                            del eye_meta['tags']
                        if filebody is not None:
                            eye_meta['filename'] = f'{filebody}_person{i}_head_eye{j}{ext}'
                        if self.return_eyes:
                            yield ImageItem(eye_image, eye_meta)

    def reset(self):
        pass
   