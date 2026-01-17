import cv2
from PIL import Image

def brg_to_rgb_to_pil(image):
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(image_rgb)

    return pil_image