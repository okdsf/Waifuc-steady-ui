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
    """
    def __init__(self,
                 # <<<--- 关键改动：移除了 output_directory --- >>>
                 esrgan_model_path: str,
                 extract_mask: bool = True,

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
        
        def _parse_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
            if not size_str: return None
            parts = re.findall(r'\d+', size_str)
            if len(parts) == 2: return int(parts[0]), int(parts[1])
            raise ValueError(f"尺寸格式无效: '{size_str}'。请输入 '宽, 高' 格式, 例如 '1024, 768'。")

        pipeline_config = {}
        if process_head:
            pipeline_config['head'] = {'enabled': True, 'target_size': _parse_size(head_target_size), 'downscale_threshold': head_downscale_threshold, 'upscale_discard_threshold': head_upscale_discard_threshold}
        if process_person:
            pipeline_config['person'] = {'enabled': True, 'target_size': _parse_size(person_target_size), 'downscale_threshold': person_downscale_threshold, 'upscale_discard_threshold': person_upscale_discard_threshold}
        if process_halfbody:
            pipeline_config['halfbody'] = {'enabled': True, 'target_size': _parse_size(halfbody_target_size), 'downscale_threshold': halfbody_downscale_threshold, 'upscale_discard_threshold': halfbody_upscale_discard_threshold}

        esrgan_config = {'model_path': esrgan_model_path}
        
        # <<<--- 关键改动：调用核心动作时不再传递 output_directory --- >>>
        super().__init__(
            WaifucDirectoryPipelineAction,
            esrgan_config=esrgan_config,
            pipeline_config=pipeline_config,
            extract_mask=extract_mask
        )