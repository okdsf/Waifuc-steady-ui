"""
pipeline_actions.py - 封装和定义完整的、自成一体的处理流水线动作
"""
import re
from typing import List, Dict, Any, Optional, Tuple

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
    然后在内部完成 分割->分类->独立放大->独立裁剪 的全过程，
    并直接将最终结果分门别类地保存到用户指定的输出目录中。

    [重要]
    在工作流中，应将此 Action 作为最后一个步骤。
    它之后不应再有任何其他 Action 或 Exporter (如 SaveExporter)。
    """
    def __init__(self,
                 esrgan_model_path: str,
                 # --- 分支处理控制 (Branch Processing Control) ---
                 process_head: bool = False,
                 process_person: bool = False,
                 process_halfbody: bool = False,

                 # --- 各分支的目标尺寸 (Target Sizes for each Branch) ---
                 head_target_size: Optional[str] = None,
                 person_target_size: Optional[str] = None,
                 halfbody_target_size: Optional[str] = None):
        """
        初始化整个流水线所需要的所有配置。
        
        :param esrgan_model_path: Real-ESRGAN 模型的绝对或相对路径。
        :param process_head: [复选框] 是否处理“头部”分支。
        :param process_person: [复选框] 是否处理“全身”分支。
        :param process_halfbody: [复选框] 是否处理“半身”分支。
        :param head_target_size: [文本框] 当“处理头部”勾选时，此项必填。格式为 "宽,高"，例如: 512, 512
        :param person_target_size: [文本框] 当“处理全身”勾选时，此项必填。格式为 "宽,高"，例如: 768, 1024
        :param halfbody_target_size: [文本框] 当“处理半身”勾选时，此项必填。格式为 "宽,高"，例如: 768, 768
        """
        
        def _parse_size(size_str: Optional[str]) -> Optional[Tuple[int, int]]:
            """一个辅助函数，用于将字符串 '宽,高' 解析为 (宽, 高) 元组。"""
            if not size_str:
                return None
            
            # 使用正则表达式匹配数字，允许各种分隔符和空格
            parts = re.findall(r'\d+', size_str)
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
            else:
                raise ValueError(f"尺寸格式无效: '{size_str}'。请输入 '宽, 高' 格式, 例如 '1024, 768'。")

        # 将用户输入的字符串尺寸，转换为程序需要的元组格式
        parsed_head_size = _parse_size(head_target_size)
        parsed_person_size = _parse_size(person_target_size)
        parsed_halfbody_size = _parse_size(halfbody_target_size)

        # 直接将所有用户友好的参数传递给 waifuc 核心 Action 的构造函数。
        super().__init__(
            WaifucDirectoryPipelineAction,
            esrgan_model_path=esrgan_model_path,
            process_head=process_head,
            process_person=process_person,
            process_halfbody=process_halfbody,
            head_target_size=parsed_head_size,
            person_target_size=parsed_person_size,
            halfbody_target_size=parsed_halfbody_size,
        )
