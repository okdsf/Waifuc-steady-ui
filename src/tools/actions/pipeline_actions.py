"""
pipeline_actions.py - 封装和定义完整的、自成一体的处理流水线动作
"""
import re
# <<<--- 关键修复：确保从 typing 模块导入了 List --- >>>
from typing import Dict, Any, List, Tuple, Optional

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
    支持为每个分支（head, person, halfbody）配置多个目标尺寸，
    并自动为每张图选择宽高比最匹配的尺寸进行处理。
    """
    def __init__(self,
                 esrgan_model_path: str,
                 extract_mask: bool = True,

                 # --- Head 分支配置 ---
                 process_head: bool = True,
                 # <<<--- 关键改动：将字符串参数拆分为独立的尺寸输入框 --- >>>
                 head_target_size_1: str = '1024, 1024',
                 head_target_size_2: str = '768, 768',
                 head_target_size_3: str = '512, 512',
                 head_downscale_threshold: float = 1.5,
                 head_upscale_discard_threshold: float = 4.0,

                 # --- Person 分支配置 ---
                 process_person: bool = True,
                 person_target_size_1: str = '1056, 1536', # 2:3
                 person_target_size_2: str = '1536, 1056', # 3:2
                 person_target_size_3: str = '1280, 1280', # 1:1
                 person_downscale_threshold: float = 1.2,
                 person_upscale_discard_threshold: float = 5.0,

                 # --- Halfbody 分支配置 ---
                 process_halfbody: bool = True,
                 halfbody_target_size_1: str = '768, 1024', # 3:4
                 halfbody_target_size_2: str = '1024, 768', # 4:3
                 halfbody_target_size_3: str = '1024, 1024', # 1:1
                 halfbody_downscale_threshold: float = 1.3,
                 halfbody_upscale_discard_threshold: float = 4.5,
                 ):
        
        # <<<--- 关键改动：简化解析逻辑以适应独立的尺寸输入 --- >>>
        def _parse_single_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
            """解析单个尺寸字符串，例如 '1024, 768'"""
            if not size_str or not size_str.strip():
                return None
            
            parts = re.findall(r'\d+', size_str)
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
            raise ValueError(f"尺寸格式无效: '{size_str}'。请输入 '宽, 高' 格式。")

        def _collect_sizes(*sizes_str: str) -> List[Tuple[int, int]]:
            """收集所有有效的尺寸输入"""
            collected_sizes = []
            for s in sizes_str:
                parsed_size = _parse_single_size(s)
                if parsed_size:
                    collected_sizes.append(parsed_size)
            return collected_sizes

        pipeline_config = {}
        # <<<--- 关键改动：使用新的尺寸收集方式构建配置 --- >>>
        if process_head:
            pipeline_config['head'] = {
                'enabled': True, 
                'target_sizes': _collect_sizes(head_target_size_1, head_target_size_2, head_target_size_3), 
                'downscale_threshold': head_downscale_threshold, 
                'upscale_discard_threshold': head_upscale_discard_threshold
            }
        if process_person:
            pipeline_config['person'] = {
                'enabled': True, 
                'target_sizes': _collect_sizes(person_target_size_1, person_target_size_2, person_target_size_3), 
                'downscale_threshold': person_downscale_threshold, 
                'upscale_discard_threshold': person_upscale_discard_threshold
            }
        if process_halfbody:
            pipeline_config['halfbody'] = {
                'enabled': True, 
                'target_sizes': _collect_sizes(halfbody_target_size_1, halfbody_target_size_2, halfbody_target_size_3), 
                'downscale_threshold': halfbody_downscale_threshold, 
                'upscale_discard_threshold': halfbody_upscale_discard_threshold
            }

        esrgan_config = {'model_path': esrgan_model_path}
        
        super().__init__(
            WaifucDirectoryPipelineAction,
            esrgan_config=esrgan_config,
            pipeline_config=pipeline_config,
            extract_mask=extract_mask
        )

