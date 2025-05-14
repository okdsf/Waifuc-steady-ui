from waifuc.action.base import ProcessAction
from waifuc.model import ImageItem
from PIL import Image
from imgutils.segment import segment_rgba_with_isnetis

class SmartCropAction(ProcessAction):
    def __init__(self, width=1024, height=1351):
        self.width = width
        self.height = height

    def process(self, item):
        # 类型检查：支持 ImageItem 和 PIL.Image
        if isinstance(item, ImageItem):
            image = item.image
            meta = item.meta
        elif isinstance(item, Image.Image):
            image = item.convert('RGB')
            meta = {}  # 默认元数据为空
        else:
            raise TypeError(f"Expected ImageItem or PIL.Image, got {type(item)}")

        # 获取带掩码的 RGBA 图像，仅用于定位角色区域
        mask_np, rgba_image = segment_rgba_with_isnetis(image)
        # 提取 alpha 通道作为掩码
        mask = rgba_image.split()[3]
        # 获取边界框
        bbox = mask.getbbox()
        if bbox is None:
            # 未找到角色，返回原图（统一封装为 ImageItem）
            return ImageItem(image, meta)
        left, top, right, bottom = bbox
        # 计算中心，优先保留上半身
        center_x = (left + right) / 2
        center_y = top + (bottom - top) / 3  # 中心点设为边界框上1/3处
        # 计算裁剪框
        crop_left = int(center_x - self.width / 2)
        crop_top = int(center_y - self.height / 2)
        crop_right = crop_left + self.width
        crop_bottom = crop_top + self.height
        # 调整裁剪框到图像范围内
        img_width, img_height = image.size
        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(img_width, crop_right)
        crop_bottom = min(img_height, crop_bottom)
        # 裁剪原图（RGB）
        cropped = image.crop((crop_left, crop_top, crop_right, crop_bottom))
        # 检查裁剪后尺寸
        cropped_width = crop_right - crop_left
        cropped_height = crop_bottom - crop_top
        # 如果裁剪尺寸小于目标尺寸，触发白色背景填充
        if cropped_width < self.width or cropped_height < self.height:
            padded = Image.new('RGB', (self.width, self.height), (255, 255, 255))
            paste_left = (self.width - cropped_width) // 2
            paste_top = (self.height - cropped_height) // 2
            padded.paste(cropped, (paste_left, paste_top))
            cropped = padded
        # 统一返回 ImageItem
        return ImageItem(cropped, meta)