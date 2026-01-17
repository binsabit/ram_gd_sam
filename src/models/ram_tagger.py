from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
import torch
import cv2
from PIL import Image
import numpy as np

class RAMImageTagger:

    def __init__(self, model_path='weights/ram_plus_swin_large_14m.pth', device=None):
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = ram_plus(pretrained=model_path, image_size=384, vit='swin_l')
        self.model.eval()
        self.model = self.model.to(self.device)

        self.transform = get_transform(image_size=384)

    def recognize(self, image):
        
        
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            result = inference(image_tensor, self.model)

        if isinstance(result, tuple):
            tags = result[0]  
        else:
            tags = result

        tag_list = [tag.strip() for tag in tags.split(',') if tag.strip()]

        return tag_list