import io
import os
import cairosvg
import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np
from scipy.ndimage import convolve

Image.MAX_IMAGE_PIXELS = None

def remove_room_classes(svg_file):
    tree = ET.parse(svg_file)
    root = tree.getroot()

    namespaces = {'svg': 'http://www.w3.org/2000/svg'}

    for tag in ['Room', 'Door']:
        for elem in root.findall(f'.//*[@class="{tag}"]', namespaces):
            root.remove(elem)

    modified_svg = ET.tostring(root, encoding='unicode').encode('utf-8')

    return modified_svg

def crop_transparent(image):
    image = image.convert("RGBA")
    data = image.getdata()

    non_transparent_pixels = [(x, y) for y in range(image.height) for x in range(image.width) if data[y * image.width + x][3] != 0]

    if not non_transparent_pixels:
        return image

    min_x = min(x for x, y in non_transparent_pixels)
    min_y = min(y for x, y in non_transparent_pixels)
    max_x = max(x for x, y in non_transparent_pixels)
    max_y = max(y for x, y in non_transparent_pixels)

    crop_box = (min_x, min_y, max_x + 1, max_y + 1)
    cropped_image = image.crop(crop_box)
    
    return cropped_image

def pixellate_image(image):
    # Convert image to binary first
    threshold = 1
    binary_image = image.convert("L").point(lambda x: 255 if x > threshold else 0, mode='1')

    # Resize image to make it pixellated
    resize_factor = 10
    small_image = binary_image.resize(
        (binary_image.width // resize_factor, binary_image.height // resize_factor), 
        Image.NEAREST
    )

    return small_image

kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])

def thicken_lines(binary_image, kernel):
    image_array = np.array(binary_image)

    convolved = convolve(image_array, kernel, mode='constant', cval=0)
    thickened_image = (convolved > 0).astype(np.uint8) * 255

    return Image.fromarray(thickened_image, mode='L')

source_folder = f"{os.getcwd()}/ImagesGT"
output_folder = f"{os.getcwd()}/data"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(source_folder):
    if not filename.endswith('.svg'):
        continue

    png_data = cairosvg.svg2png(bytestring=remove_room_classes(f"{source_folder}/{filename}"), output_width=12000, output_height=12000)

    cropped_image = crop_transparent(Image.open(io.BytesIO(png_data)))

    pixellated_image = pixellate_image(cropped_image)

    thickened_image = thicken_lines(pixellated_image, kernel)

    thickened_image.save(f'data/{filename[:-4]}.png')
