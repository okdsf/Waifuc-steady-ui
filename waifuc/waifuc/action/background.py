from imgutils.segment import segment_rgba_with_isnetis

from .base import ProcessAction
from ..model import ImageItem
from PIL import Image

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
    

class PersonRemovalAction(ProcessAction): # Or you can modify your existing BackgroundRemovalAction
    def process(self, item):
        if isinstance(item, ImageItem):
            original_image = item.image
            meta = item.meta
        elif isinstance(item, Image.Image):
            original_image = item
            meta = {}
        else:
            raise TypeError(f"Expected ImageItem or PIL.Image, got {type(item)} [cite: 1]")
        image_for_segmentation = original_image.convert('RGB')

        _, person_with_alpha = segment_rgba_with_isnetis(image_for_segmentation)

        # 3. Extract the alpha mask. This mask typically has high values (e.g., 255)
        #    for the person and low values (e.g., 0) for the background.
        if person_with_alpha.mode != 'RGBA':
            # This scenario might indicate an issue with segmentation or an unexpected format.
            raise ValueError("Segmentation did not return an RGBA image as anticipated.")
        
        person_mask = person_with_alpha.split()[3] # This is an 'L' mode image (grayscale alpha mask)

        # 4. Prepare the output image from the original.
        #    We will fill the person's area in this image.
        output_image = original_image.convert('RGB')

        # 5. Define the fill color (e.g., white) and create an image with this color.
        fill_color = (255, 255, 255) # White, like the background in your original script [cite: 1]
        filler_image = Image.new('RGB', output_image.size, fill_color)
        output_image.paste(filler_image, mask=person_mask)

        return ImageItem(output_image, meta)
