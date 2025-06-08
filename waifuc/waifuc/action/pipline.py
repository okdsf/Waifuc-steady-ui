"""
waifuc/action/pipeline.py - 核心功能模块
"""
import os
import shutil
import tempfile
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Iterable, Iterator, Optional, Tuple

# 从 waifuc 框架导入基础模块
from .base import TerminalAction
from ..model import ImageItem

# 从当前 action 目录导入您已经编写和验证过的功能模块
from .split import ThreeStageSplitAction
from .esrgan import ESRGANAction
from .crop import SmartCropAction

class DirectoryPipelineAction(TerminalAction):
    """
    一个自成一体的、作为“终点站”的自动化图像处理流水线。
    (这个类的核心实现，现在与封装类中的扁平化参数相匹配)
    """

    def __init__(self,
                 esrgan_model_path: str,
                 # --- 分支处理控制 (Branch Processing Control) ---
                 process_head: bool = False,
                 process_person: bool = False,
                 process_halfbody: bool = False,

                 # --- 各分支的目标尺寸 (Target Sizes for each Branch) ---
                 head_target_size: Optional[Tuple[int, int]] = None,
                 person_target_size: Optional[Tuple[int, int]] = None,
                 halfbody_target_size: Optional[Tuple[int, int]] = None):
        
        # 将扁平化的参数在内部转换为统一的 pipeline_config 字典列表
        self.pipeline_config = []
        if process_head:
            if not head_target_size: raise ValueError("如果 'process_head' 为 True, 则 'head_target_size' 必须被指定。")
            self.pipeline_config.append({'type': 'head', 'target_size': head_target_size})
        
        if process_person:
            if not person_target_size: raise ValueError("如果 'process_person' 为 True, 则 'person_target_size' 必须被指定。")
            self.pipeline_config.append({'type': 'person', 'target_size': person_target_size})

        if process_halfbody:
            if not halfbody_target_size: raise ValueError("如果 'process_halfbody' 为 True, 则 'halfbody_target_size' 必须被指定。")
            self.pipeline_config.append({'type': 'halfbody', 'target_size': halfbody_target_size})

        if not self.pipeline_config:
            raise ValueError("至少必须启用一个处理分支 (process_head, process_person, 或 process_halfbody)。")
            
        self.esrgan_model_path = esrgan_model_path
        
        # 这个属性将由 Engine 在运行时设置
        self.output_directory = None
        self.temp_root = None

    def _calculate_upscale_factor(self, image_folder: str, target_size: tuple) -> float:
        # (核心算法逻辑不变)
        image_areas = []
        files_to_scan = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
        if not files_to_scan: return 1.0
        
        for filename in files_to_scan:
            filepath = os.path.join(image_folder, filename)
            try:
                with Image.open(filepath) as img:
                    w, h = img.size
                    if w > 0 and h > 0:
                        image_areas.append({'area': w * h, 'size': (w, h)})
            except Exception: continue
        
        if not image_areas: return 1.0
        image_areas.sort(key=lambda x: x['area'])
        median_image_data = image_areas[len(image_areas) // 2]
        median_w, median_h = median_image_data['size']
        target_w, target_h = target_size

        if median_w >= target_w and median_h >= target_h: return 1.0
        
        ratio_w = target_w / median_w if median_w > 0 else float('inf')
        ratio_h = target_h / median_h if median_h > 0 else float('inf')
        factor = max(ratio_w, ratio_h)
        return np.ceil(factor * 10) / 10 if factor != float('inf') else 1.0

    def iter_from(self, iter_: Iterable[ImageItem]) -> Iterator[ImageItem]:
        if self.output_directory is None:
            raise RuntimeError("DirectoryPipelineAction 错误: output_directory 未被 WorkflowEngine 正确设置。")

        self.temp_root = tempfile.mkdtemp()
        source_images_dir = os.path.join(self.temp_root, 'source_input')
        processing_dir = os.path.join(self.temp_root, 'processing')
        os.makedirs(source_images_dir, exist_ok=True)
        os.makedirs(processing_dir, exist_ok=True)

        try:
            print("--- [阶段 0/5] 接收并暂存输入流... ---")
            for item in tqdm(iter_, desc="Caching input stream"):
                unique_filename = f"{os.path.splitext(item.meta.get('filename', 'unknown'))[0]}_{hash(item.meta.get('filename', 'unknown'))}.png"
                item.image.save(os.path.join(source_images_dir, unique_filename))
            
            active_branches = {}
            for cfg in self.pipeline_config:
                branch_type = cfg['type']
                active_branches[branch_type] = {
                    'processing_path': os.path.join(processing_dir, f"{branch_type}s"),
                    'final_path': os.path.join(self.output_directory, f"{branch_type}s"),
                    'target_size': cfg['target_size']
                }
                os.makedirs(active_branches[branch_type]['processing_path'], exist_ok=True)
                os.makedirs(active_branches[branch_type]['final_path'], exist_ok=True)
            
            print("--- [阶段 1/5] 执行图像分割... ---")
            split_flags = {key: True for key in active_branches.keys()}
            splitter = ThreeStageSplitAction(
                return_person=split_flags.get('person', False),
                return_halfbody=split_flags.get('halfbody', False),
                return_head=split_flags.get('head', False)
            )
            source_files = [f for f in os.listdir(source_images_dir) if os.path.isfile(os.path.join(source_images_dir, f))]
            for filename in tqdm(source_files, desc="分割原图"):
                try:
                    image = Image.open(os.path.join(source_images_dir, filename))
                    for i, item in enumerate(splitter.iter(ImageItem(image, {'filename': filename}))):
                        crop_type = item.meta.get('crop', {}).get('type')
                        if crop_type in active_branches:
                            item.image.save(os.path.join(active_branches[crop_type]['processing_path'], f"{os.path.splitext(filename)[0]}_{i}.png"))
                except Exception as e: print(f"警告: 分割文件 {filename} 时出错: {e}")

            for branch_type, branch_data in active_branches.items():
                folder_path = branch_data['processing_path']
                if not os.listdir(folder_path): continue
                print(f"\n--- 开始处理 [{branch_type.capitalize()}] 分支 ---")
                factor = self._calculate_upscale_factor(folder_path, branch_data['target_size'])
                if factor > 1.0:
                    upscaler = ESRGANAction(scale=factor, model_path=self.esrgan_model_path)
                    self._apply_action_to_folder(folder_path, upscaler, f"放大 {branch_type.capitalize()}")
                cropper = SmartCropAction(width=branch_data['target_size'][0], height=branch_data['target_size'][1])
                self._apply_action_to_folder(folder_path, cropper, f"裁剪 {branch_type.capitalize()}")

            print("\n--- [阶段 5/5] ...正在将结果保存到最终目录... ---")
            for branch_type, branch_data in active_branches.items():
                source_folder, dest_folder = branch_data['processing_path'], branch_data['final_path']
                if os.path.exists(source_folder):
                    for filename in tqdm(os.listdir(source_folder), desc=f"Saving {branch_type.capitalize()}s"):
                        shutil.copy2(os.path.join(source_folder, filename), os.path.join(dest_folder, filename))
        finally:
            if self.temp_root and os.path.exists(self.temp_root): shutil.rmtree(self.temp_root)
        
        yield from []

    def _apply_action_to_folder(self, folder_path, action, desc):
        temp_dir = folder_path + '_temp'
        os.makedirs(temp_dir, exist_ok=True)
        file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        for filename in tqdm(file_list, desc=desc):
            try:
                with Image.open(os.path.join(folder_path, filename)) as img:
                    processed_item = action.process(ImageItem(img))
                    processed_item.image.save(os.path.join(temp_dir, filename))
            except Exception as e: print(f"警告: 在 '{desc}' 阶段处理文件 {filename} 时出错: {e}")
        shutil.rmtree(folder_path)
        os.rename(temp_dir, folder_path)

    def reset(self):
        pass