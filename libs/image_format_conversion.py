import base64

import cv2
from PIL import Image
import numpy as np

def convert_to_pillow(opencv_img: cv2.typing.MatLike) -> Image:
    return Image.fromarray(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB))


def convert_to_opencv(pil_img: Image) -> cv2.typing.MatLike:
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def convert_opencv_to_base64(opencv_img: cv2.typing.MatLike, image_format='jpeg') -> str:
    # Encode the image to a specific format (e.g., JPEG or PNG)
    success, encoded_image = cv2.imencode(f'.{image_format}', opencv_img)

    # Ensure the encoding was successful
    if not success:
        raise ValueError("Image encoding failed")

    # Convert the encoded image to base64
    base64_string = base64.b64encode(encoded_image).decode('utf-8')

    # Create the data URI for embedding directly in HTML
    data_uri = f"data:image/{image_format};base64,{base64_string}"

    return data_uri