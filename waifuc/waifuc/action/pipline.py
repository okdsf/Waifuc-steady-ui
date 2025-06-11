import os
import shutil
import tempfile
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, Iterable, Iterator

from .base import TerminalAction, BaseAction, ProcessAction
from ..model import ImageItem
from .split import ThreeStageSplitAction
from .preprocess import PreprocessAction
from .framing import FramingCropAction

# 默认配置，如果用户没有提供，则使用此配置
DEFAULT_PIPELINE_SETTINGS = {
    'head': {
        'enabled': True,
        'target_size': (512, 512),
        'downscale_threshold': 1.5,
        'upscale_discard_threshold': 4.0,
    },
    'person': {
        'enabled': True,
        'target_size': (768, 1152),
        'downscale_threshold': 1.2,
        'upscale_discard_threshold': 5.0,
    },
    'halfbody': {
        'enabled': False,
        'target_size': (768, 768),
        'downscale_threshold': 1.3,
        'upscale_discard_threshold': 4.5,
    }
}

class DirectoryPipelineAction(TerminalAction):
    """
    一个自成一体的、作为“终点站”的自动化图像处理流水线。
    它使用结构化的配置，并按顺序调用分割、预处理和构图裁剪动作。
    """

    def __init__(self, esrgan_model_path: str, 
                 pipeline_config: Dict[str, Dict[str, Any]] = None):
        
        self.pipeline_config = pipeline_config or DEFAULT_PIPELINE_SETTINGS
        self.esrgan_model_path = esrgan_model_path
        
        if not any(cfg.get('enabled', False) for cfg in self.pipeline_config.values()):
            raise ValueError("至少必须启用一个处理分支 (e.g., 'head', 'person')。")
            
        self.output_directory = None
        self.temp_root = None

    def _apply_actions_to_folder(self, folder_path: str, actions: [BaseAction], desc: str):
        """通用函数，对文件夹中的所有图片按顺序应用一系列动作"""
        if not os.path.exists(folder_path) or not os.listdir(folder_path):
            return

        current_path = folder_path
        for i, action in enumerate(actions):
            next_path = f"{folder_path}_step_{i+1}"
            os.makedirs(next_path, exist_ok=True)
            
            pbar = tqdm(os.listdir(current_path), desc=f"{desc} (步骤 {i+1}/{len(actions)})")
            for filename in pbar:
                try:
                    item = ImageItem.load_from_image(os.path.join(current_path, filename))
                    
                    if isinstance(action, ProcessAction):
                        processed_item = action.process(item)
                        if processed_item:
                            processed_item.save(os.path.join(next_path, filename))
                    else:
                        for j, processed_item in enumerate(action.iter(item)):
                            fname, ext = os.path.splitext(filename)
                            new_filename = f"{fname}_{j}{ext}"
                            processed_item.save(os.path.join(next_path, new_filename))
                except Exception as e:
                    print(f"警告: 在处理文件 {filename} 时出错: {e}")
            
            if current_path != folder_path:
                shutil.rmtree(current_path)
            current_path = next_path
        
        if os.path.exists(folder_path):
             shutil.rmtree(folder_path)
        os.rename(current_path, folder_path)


    def iter_from(self, iter_: Iterable[ImageItem]) -> Iterator[ImageItem]:
        if self.output_directory is None:
            raise RuntimeError("错误: output_directory 未被正确设置。")

        self.temp_root = tempfile.mkdtemp()
        source_images_dir = os.path.join(self.temp_root, '00_source_input')
        split_output_dir = os.path.join(self.temp_root, '01_split_output')
        os.makedirs(source_images_dir, exist_ok=True)
        os.makedirs(split_output_dir, exist_ok=True)

        try:
            print("--- [阶段 0/4] 接收并暂存输入流... ---")
            for item in tqdm(iter_, desc="缓存输入流"):
                unique_filename = f"{os.path.splitext(item.meta.get('filename', 'unknown'))[0]}_{hash(item.meta.get('filename', 'unknown'))}.png"
                item.save(os.path.join(source_images_dir, unique_filename))
            
            print("\n--- [阶段 1/4] 执行图像分割... ---")
            enabled_types = [t for t, cfg in self.pipeline_config.items() if cfg.get('enabled')]
            splitter = ThreeStageSplitAction(
                return_person='person' in enabled_types,
                return_halfbody='halfbody' in enabled_types,
                return_head='head' in enabled_types
            )
            for filename in tqdm(os.listdir(source_images_dir), desc="分割原图"):
                try:
                    item = ImageItem.load_from_image(os.path.join(source_images_dir, filename))
                    for i, result_item in enumerate(splitter.iter(item)):
                        base_detection = result_item.meta.get('base_detection', {})
                        # 优先使用 base_detection 的类型，其次是 crop 的
                        crop_type = base_detection.get('type') or result_item.meta.get('crop', {}).get('type', 'unknown')
                        if crop_type in enabled_types:
                            branch_dir = os.path.join(split_output_dir, crop_type)
                            os.makedirs(branch_dir, exist_ok=True)
                            fname, ext = os.path.splitext(filename)
                            result_item.save(os.path.join(branch_dir, f"{fname}_{crop_type}_{i}{ext}"))
                except Exception as e:
                    print(f"警告: 分割文件 {filename} 时出错: {e}")

            print("\n--- [阶段 2/4 & 3/4] 对各分支进行预处理和构图裁剪... ---")
            for branch_type, branch_cfg in self.pipeline_config.items():
                if not branch_cfg.get('enabled'): continue
                
                branch_path = os.path.join(split_output_dir, branch_type)
                print(f"\n--- 开始处理 [{branch_type.capitalize()}] 分支 ---")

                branch_actions = [
                    PreprocessAction(
                        target_size=branch_cfg['target_size'],
                        downscale_threshold=branch_cfg['downscale_threshold'],
                        upscale_discard_threshold=branch_cfg['upscale_discard_threshold'],
                        esrgan_model_path=self.esrgan_model_path
                    ),
                    FramingCropAction(size=branch_cfg['target_size'])
                ]
                
                self._apply_actions_to_folder(branch_path, branch_actions, f"处理 {branch_type.capitalize()}")

            print("\n--- [阶段 4/4] 保存结果到最终目录... ---")
            for branch_type, branch_cfg in self.pipeline_config.items():
                if not branch_cfg.get('enabled'): continue
                source_folder = os.path.join(split_output_dir, branch_type)
                dest_folder = os.path.join(self.output_directory, branch_type)
                os.makedirs(dest_folder, exist_ok=True)
                if os.path.exists(source_folder):
                    for filename in tqdm(os.listdir(source_folder), desc=f"保存 {branch_type.capitalize()}s"):
                        source_file = os.path.join(source_folder, filename)
                        dest_file = os.path.join(dest_folder, filename)
                        # 读取 item 并使用 no_meta=True 来保存，确保不产生json文件
                        try:
                           final_item = ImageItem.load_from_image(source_file)
                           final_item.save(dest_file, no_meta=True)
                        except Exception as e:
                           print(f"警告: 保存最终文件 {filename} 时失败: {e}。将直接复制。")
                           shutil.copy2(source_file, dest_file)
        finally:
            if self.temp_root and os.path.exists(self.temp_root):
                shutil.rmtree(self.temp_root)
        
        yield from []

    def reset(self):
        pass
