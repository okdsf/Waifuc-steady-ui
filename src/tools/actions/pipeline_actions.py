"""
pipeline_actions.py - 封装和定义完整的、自成一体的处理流水线动作
"""
import re
from typing import Dict, Any, Optional, Tuple

# 导入您封装系统的基础模块
from .waifuc_actions import WaifucActionWrapper
# 导入我们编写的核心功能模块
# 严格遵循项目导入规范，使用 as 别名
from waifuc.action import (
    DirectoryPipelineAction as WaifucDirectoryPipelineAction,
)


class DirectoryPipelineActionWrapper(WaifucActionWrapper):
    """
    一个完整的、自动化的图像处理流水线。

    本动作是一个“终点站”，它会消费掉所有输入流，
    然后在内部完成 分割->预处理(筛选与缩放)->构图裁剪 的全过程，
    并直接将最终结果分门别类地保存到用户指定的输出目录中。
    """
    def __init__(self,
                 esrgan_model_path: str,

                 # --- Head 分支配置 ---
                 process_head: bool = True,
                 head_target_size: str = '1280, 1280',
                 head_downscale_threshold: float = 1.5,
                 head_upscale_discard_threshold: float = 4.0,

                 # --- Person 分支配置 ---
                 process_person: bool = True,
                 person_target_size: str = '1056, 1536',
                 person_downscale_threshold: float = 1.2,
                 person_upscale_discard_threshold: float = 5.0,

                 # --- Halfbody 分支配置 ---
                 process_halfbody: bool = False,
                 halfbody_target_size: str = '1056, 1536',
                 halfbody_downscale_threshold: float = 1.3,
                 halfbody_upscale_discard_threshold: float = 4.5,
                 ):
        """
        初始化整个流水线所需要的所有配置。

        :param esrgan_model_path: Real-ESRGAN 模型的绝对或相对路径。
        
        :param process_head: [复选框] 是否处理“头部”分支。
        :param head_target_size: [文本框] 头部目标尺寸，格式 "宽,高"。
        :param head_downscale_threshold: [数字输入] 头部冗余度阈值，建议范围 1.2-1.8。
        :param head_upscale_discard_threshold: [数字输入] 头部放大丢弃阈值，建议范围 3-5。

        :param process_person: [复选框] 是否处理“全身”分支。
        :param person_target_size: [文本框] 全身目标尺寸，格式 "宽,高"。
        :param person_downscale_threshold: [数字输入] 全身冗余度阈值，建议范围 1.1-1.5。
        :param person_upscale_discard_threshold: [数字输入] 全身放大丢弃阈值，建议范围 4-6。

        :param process_halfbody: [复选框] 是否处理“半身”分支。
        :param halfbody_target_size: [文本框] 半身目标尺寸，格式 "宽,高"。
        :param halfbody_downscale_threshold: [数字输入] 半身冗余度阈值，建议范围 1.2-1.6。
        :param halfbody_upscale_discard_threshold: [数字输入] 半身放大丢弃阈值，建议范围 3.5-5.5。
        """
        
        def _parse_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
            """一个辅助函数，用于将字符串 '宽,高' 解析为 (宽, 高) 元组。"""
            if not size_str: return None
            parts = re.findall(r'\d+', size_str)
            if len(parts) == 2: return int(parts[0]), int(parts[1])
            raise ValueError(f"尺寸格式无效: '{size_str}'。请输入 '宽, 高' 格式, 例如 '1024, 768'。")

        # --- 在内部构建结构化的 pipeline_config 字典 ---
        pipeline_config = {}

        if process_head:
            pipeline_config['head'] = {
                'enabled': True,
                'target_size': _parse_size(head_target_size),
                'downscale_threshold': head_downscale_threshold,
                'upscale_discard_threshold': head_upscale_discard_threshold,
            }

        if process_person:
            pipeline_config['person'] = {
                'enabled': True,
                'target_size': _parse_size(person_target_size),
                'downscale_threshold': person_downscale_threshold,
                'upscale_discard_threshold': person_upscale_discard_threshold,
            }

        if process_halfbody:
            pipeline_config['halfbody'] = {
                'enabled': True,
                'target_size': _parse_size(halfbody_target_size),
                'downscale_threshold': halfbody_downscale_threshold,
                'upscale_discard_threshold': halfbody_upscale_discard_threshold,
            }
        
        # 将组装好的配置传递给核心的 WaifucDirectoryPipelineAction
        super().__init__(
            WaifucDirectoryPipelineAction,
            esrgan_model_path=esrgan_model_path,
            pipeline_config=pipeline_config,
        )
