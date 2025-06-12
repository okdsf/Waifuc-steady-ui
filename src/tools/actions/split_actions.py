"""
split_actions.py - 图像分割相关的动作
"""
from typing import Optional, Dict
from .waifuc_actions import WaifucActionWrapper
from waifuc.action import (
    PersonSplitAction as WaifucPersonSplitAction,
    ThreeStageSplitAction as WaifucThreeStageSplitAction,
    FrameSplitAction as WaifucFrameSplitAction
)

class PersonSplitAction(WaifucActionWrapper):
    """
    检测并裁剪图像中的每个人物，生成单独的图像。
    
    参数:
        keep_original (bool): 是否保留原始图像，默认为 False。
        level (str): 检测级别，默认为 'm'。
        version (str): 检测模型版本，默认为 'v1.1'。
        conf_threshold (float): 置信度阈值，默认为 0.3。
        iou_threshold (float): IOU 阈值，默认为 0.5。
        keep_origin_tags (bool): 是否保留原始标签，默认为 False。
    """
    def __init__(self, keep_original: bool = False, level: str = 'm', version: str = 'v1.1',
                 conf_threshold: float = 0.3, iou_threshold: float = 0.5, keep_origin_tags: bool = False):
        super().__init__(WaifucPersonSplitAction, keep_original=keep_original, level=level, version=version,
                        conf_threshold=conf_threshold, iou_threshold=iou_threshold, keep_origin_tags=keep_origin_tags)

class ThreeStageSplitAction(WaifucActionWrapper):
    """
    进行全身、半身、头部的三阶段分割。
    """
    def __init__(self,
                 person_conf: Optional[Dict] = None, halfbody_conf: Optional[Dict] = None,
                 head_conf: Optional[Dict] = None, head_scale: float = 1.5,
                 split_person: bool = True, keep_origin_tags: bool = False,
                 return_person: bool = True, return_halfbody: bool = True,
                 return_head: bool = True,
                 # <<<--- 关键改动：添加了与新核心动作匹配的 extract_mask 参数 --- >>>
                 extract_mask: bool = True):
        """
        :param extract_mask: [复选框] 是否提取精细的人物掩码。这是一个高消耗操作，但能为后续步骤提供最精确的几何信息。
        ... (其他参数文档不变) ...
        """
        # <<<--- 关键改动：移除了过时的 eye 相关参数，添加了 extract_mask --- >>>
        super().__init__(
            WaifucThreeStageSplitAction,
            person_conf=person_conf,
            halfbody_conf=halfbody_conf,
            head_conf=head_conf,
            head_scale=head_scale,
            split_person=split_person,
            keep_origin_tags=keep_origin_tags,
            return_person=return_person,
            return_halfbody=return_halfbody,
            return_head=return_head,
            extract_mask=extract_mask  # 传递新参数
        )
        
class FrameSplitAction(WaifucActionWrapper):
    """
    将多帧图像（如GIF）分割成单帧。
    """
    def __init__(self):
        super().__init__(WaifucFrameSplitAction)