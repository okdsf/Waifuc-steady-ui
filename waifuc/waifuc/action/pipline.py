import os
import shutil
import tempfile
import logging
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, Iterable, Iterator, List, Tuple

# 恢复原有的相对导入
from .base import TerminalAction, BaseAction, ProcessAction
from ..model import ImageItem
from .split import ThreeStageSplitAction
from .preprocess import PreprocessAction
from .framing import FramingCropAction


# 默认配置已更新，以支持多尺寸自适应
# 每个分支现在使用 'target_sizes' 列表代替单一的 'target_size'
DEFAULT_PIPELINE_SETTINGS = {
    'head': {
        'enabled': True, 'target_sizes': [(512, 512), (768, 768), (1024, 1024)],
        'downscale_threshold': 1.5, 'upscale_discard_threshold': 4.0,
    },
    'person': {
        'enabled': True, 'target_sizes': [(1280, 1280), (1056, 1536), (1536, 1056)], # 1:1, 2:3, 3:2
        'downscale_threshold': 1.2, 'upscale_discard_threshold': 5.0,
    },
    'halfbody': {
        'enabled': True, 'target_sizes': [(1024, 1024), (768, 1024), (1024, 768)], # 1:1, 3:4, 4:3
        'downscale_threshold': 1.3, 'upscale_discard_threshold': 4.5,
    }
}

VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}

class DirectoryPipelineAction(TerminalAction):
    """
    一个自成一体的、作为“终点站”的自动化图像处理流水线。
    支持根据图像原始比例动态选择最合适的目标尺寸进行处理。
    """
    def __init__(self,
                 pipeline_config: Dict[str, Dict[str, Any]] = None,
                 extract_mask: bool = True,
                 esrgan_config: Dict[str, Any] = None, 
                 **kwargs):
        
        self.pipeline_config = pipeline_config or DEFAULT_PIPELINE_SETTINGS
        self.extract_mask = extract_mask
        
        self.output_directory = None 
        
        final_esrgan_config = esrgan_config
        if 'esrgan_model_path' in kwargs:
            logging.warning("The 'esrgan_model_path' argument is deprecated. Please use 'esrgan_config' dictionary.")
            if final_esrgan_config is None:
                final_esrgan_config = {'model_path': kwargs['esrgan_model_path']}
        
        if final_esrgan_config is None:
            raise TypeError("DirectoryPipelineAction requires the 'esrgan_config' dictionary argument.")
        
        self.esrgan_config = final_esrgan_config
        
        if not any(cfg.get('enabled', False) for cfg in self.pipeline_config.values()):
            raise ValueError("Pipeline configuration error: At least one branch ('head', 'person', etc.) must be enabled.")
            
        self.temp_root = None
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def iter_from(self, iter_: Iterable[ImageItem]) -> Iterator[ImageItem]:
        if not self.output_directory or not os.path.isdir(os.path.dirname(self.output_directory)):
             raise RuntimeError(f"Error: output_directory ('{self.output_directory}') was not correctly set by the workflow engine before execution.")
        os.makedirs(self.output_directory, exist_ok=True)


        self.temp_root = tempfile.mkdtemp()
        source_images_dir = os.path.join(self.temp_root, '00_source_input')
        split_output_dir = os.path.join(self.temp_root, '01_split_output')
        os.makedirs(source_images_dir, exist_ok=True)
        os.makedirs(split_output_dir, exist_ok=True)

        try:
            logging.info("--- [Phase 0/4] Caching input stream... ---")
            for item in tqdm(iter_, desc="Caching input"):
                orig_filename = item.meta.get('filename', f'unknown_{hash(item.image.tobytes())}.png')
                item.save(os.path.join(source_images_dir, os.path.basename(orig_filename)))
            
            logging.info("\n--- [Phase 1/4] Performing image splitting... ---")
            enabled_types = [t for t, cfg in self.pipeline_config.items() if cfg.get('enabled')]
            splitter = ThreeStageSplitAction(
                extract_mask=self.extract_mask,
                return_person='person' in enabled_types,
                return_halfbody='halfbody' in enabled_types,
                return_head='head' in enabled_types
            )
            image_files = [f for f in os.listdir(source_images_dir) if os.path.splitext(f.lower())[1] in VALID_IMAGE_EXTENSIONS]
            for filename in tqdm(image_files, desc="Splitting source images"):
                try:
                    item = ImageItem.load_from_image(os.path.join(source_images_dir, filename))
                    for result_item in splitter.iter(item):
                        branch_type = result_item.meta.get('branch_type', 'unknown')
                        if branch_type in enabled_types:
                            branch_dir = os.path.join(split_output_dir, branch_type)
                            os.makedirs(branch_dir, exist_ok=True)
                            new_filename = os.path.basename(result_item.meta.get('filename', f"{filename}_{branch_type}.png"))
                            result_item.save(os.path.join(branch_dir, new_filename))
                except Exception as e:
                    logging.error(f"Error splitting file {filename}: {e}", exc_info=True)

            logging.info("\n--- [Phase 2 & 3] Preprocessing and framing each branch with adaptive sizing... ---")
            for branch_type, branch_cfg in self.pipeline_config.items():
                if not branch_cfg.get('enabled'): continue
                branch_path = os.path.join(split_output_dir, branch_type)
                if not os.path.exists(branch_path): continue
                logging.info(f"\n--- Processing branch: [{branch_type.capitalize()}] ---")
                
                self._apply_actions_to_folder_adaptively(
                    branch_path, 
                    branch_cfg, 
                    f"Processing {branch_type.capitalize()}"
                )

            logging.info("\n--- [Phase 4/4] Saving final results... ---")
            for branch_type in enabled_types:
                source_folder = os.path.join(split_output_dir, branch_type)
                dest_folder = os.path.join(self.output_directory, branch_type)
                os.makedirs(dest_folder, exist_ok=True)
                if os.path.exists(source_folder):
                    for filename in tqdm(os.listdir(source_folder), desc=f"Saving {branch_type.capitalize()}s"):
                        shutil.copy2(os.path.join(source_folder, filename), os.path.join(dest_folder, filename))
        finally:
            if self.temp_root and os.path.exists(self.temp_root):
                logging.info(f"Cleaning up temporary directory: {self.temp_root}")
                shutil.rmtree(self.temp_root)
        
        yield from []

    def _apply_actions_to_folder_adaptively(self, folder_path: str, branch_cfg: Dict, desc: str):
        if not os.path.exists(folder_path) or not os.listdir(folder_path): return
        
        dest_path = f"{folder_path}_processed"
        os.makedirs(dest_path, exist_ok=True)
        
        image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f.lower())[1] in VALID_IMAGE_EXTENSIONS]
        pbar = tqdm(image_files, desc=desc)

        target_sizes = branch_cfg['target_sizes']
        if not target_sizes:
            logging.warning(f"No target_sizes defined for branch, skipping processing.")
            return

        for filename in pbar:
            try:
                item_path = os.path.join(folder_path, filename)
                item = ImageItem.load_from_image(item_path)

                if not item.image: continue
                img_w, img_h = item.image.size
                if img_h == 0 or img_w == 0: continue
                
                # --- 动态尺寸选择逻辑 ---
                img_ratio = img_w / img_h
                best_target_size = None
                min_ratio_diff = float('inf')

                for ts in target_sizes:
                    ts_w, ts_h = ts
                    if ts_h == 0 or ts_w == 0: continue
                    target_ratio = ts_w / ts_h
                    ratio_diff = abs(img_ratio - target_ratio)
                    
                    if ratio_diff < min_ratio_diff:
                        min_ratio_diff = ratio_diff
                        best_target_size = ts
                # --- 选择结束 ---

                if not best_target_size: continue
                pbar.set_postfix_str(f"Best fit for {filename}: {best_target_size}")

                # 使用选择的最佳尺寸即时创建处理动作
                preprocess_action = PreprocessAction(
                    target_size=best_target_size,
                    downscale_threshold=branch_cfg['downscale_threshold'],
                    upscale_discard_threshold=branch_cfg['upscale_discard_threshold'],
                    esrgan=self.esrgan_config
                )
                
                framing_action = FramingCropAction(size=best_target_size)

                # 依次应用处理动作
                processed_item = preprocess_action.process(item)
                if processed_item:
                    final_item = framing_action.process(processed_item)
                    if final_item:
                        final_item.save(os.path.join(dest_path, filename))

            except Exception as e:
                logging.error(f"Error adaptively processing file {filename}: {e}", exc_info=True)
        
        shutil.rmtree(folder_path)
        os.rename(dest_path, folder_path)

    def reset(self):
        pass
