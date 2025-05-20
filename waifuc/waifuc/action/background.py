from imgutils.segment import segment_rgba_with_isnetis

from .base import ProcessAction
from ..model import ImageItem

class BackgroundRemovalAction(ProcessAction):
    def process(self, item):
        if isinstance(item, ImageItem):
            image = item.image
            meta = item.meta
        elif isinstance(item, Image.Image):
            image = item.convert('RGB')
            meta = {}
        else:
            raise TypeError(f"Expected ImageItem or PIL.Image, got {type(item)}")

        _, rgba_image = segment_rgba_with_isnetis(image)
        if rgba_image.mode == 'RGBA':
            rgb_image = Image.new('RGB', rgba_image.size, (255, 255, 255))
            rgb_image.paste(rgba_image, mask=rgba_image.split()[3])
            image = rgb_image
        else:
            image = rgba_image

        return ImageItem(image, meta)