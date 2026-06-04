__copyright__ = \
    """
    Copyright (C) 2024 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the MIT License.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 18, 2024
    """
__author__ = "Alexandre Delplanque"
__license__ = "MIT License"
__version__ = "0.2.1"


import PIL

from PIL import Image, ImageDraw, ImageFont

__all__ = ['draw_points', 'draw_boxes', 'draw_text']

def draw_points(
    image: PIL.Image.Image, 
    points: list, 
    color: str = 'red',
    size: int = 4
    ) -> PIL.Image.Image:

    draw = ImageDraw.Draw(image)

    for p in points:
        e = [
            p[1] - (size // 2), p[0] - (size // 2),
            p[1] + (size // 2), p[0] + (size // 2)
            ]

        draw.ellipse(e, fill=color, outline='black')
  
    return image

def draw_boxes(
    image: PIL.Image.Image, 
    boxes: list, 
    color: str = 'red',
    width: int = 1
    ) -> PIL.Image.Image:

    draw = ImageDraw.Draw(image)

    for b in boxes:
        draw.rectangle(b, fill=None, outline=color, width=width)
  
    return image

def draw_text(
    image: PIL.Image.Image,
    text: str,
    position: tuple,
    font_size: int = 20,
    ) -> PIL.Image.Image:

    draw = ImageDraw.Draw(image)

    # Try a truetype font so font_size matters; fallback to default bitmap font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    # Compute text bounding box with fallback for older Pillow or bitmap fonts
    try:
        # Preferred (Pillow >= ~8.0 and truetype font)
        l, t, r, b = draw.textbbox(position, text, font=font)
    except Exception:
        # Fallback path (older Pillow or bitmap font without getbbox)
        w, h = draw.textsize(text, font=font)
        l, t = position
        r, b = l + w, t + h

    # White box background + black text
    draw.rectangle((l - 5, t - 5, r + 5, b + 5), fill='white')
    draw.text(position, text, font=font, fill='black')
    return image

