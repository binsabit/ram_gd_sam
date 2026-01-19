import os
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model as load_gd_model, predict as predict_gd
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class Segmentor:
    def __init__(self, 
                 gd_config="GroundingDINO_SwinT_OGC.py", 
                 gd_ckpt="groundingdino_swint_ogc.pth",
                 sam2_config="sam2.1_hiera_l.yaml",     
                 sam2_ckpt="sam2.1_hiera_large.pt",
                 device="cuda"):
        
        self.device = device
        
        # --- Load Grounding DINO ---
        print("Loading Grounding DINO...")
        # Grounding DINO handles paths correctly, so we resolve abspath for it if needed
        if not os.path.isabs(gd_config):
             gd_config = os.path.abspath(gd_config)

        self.gd_model = load_gd_model(gd_config, gd_ckpt, device=self.device)
        
        # --- Load SAM 2 ---
        print(f"Loading SAM 2 with config: {sam2_config}...")
        
        # NOTE: SAM 2 (Hydra) expects a CONFIG NAME (e.g. "sam2.1_hiera_l.yaml"), 
        # NOT a file path. It looks inside the installed sam2 python library.
        # We do NOT use os.path.abspath here.
        self.sam2_model = build_sam2(sam2_config, sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        print("Models loaded successfully.")

    def _preprocess_gd(self, image_bgr):
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pil, None)
        return image_transformed.to(self.device)

    def process_image(self, image_bgr, text_prompt, box_threshold=0.35, text_threshold=0.25):
        # 1. GROUNDING DINO STEP
        image_tensor = self._preprocess_gd(image_bgr)
        
        boxes, logits, phrases = predict_gd(
            model=self.gd_model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )

        if len(boxes) == 0:
            return sv.Detections.empty()

        # 2. CONVERT BOXES (Normalized cxcywh -> Pixel xyxy)
        h, w, _ = image_bgr.shape
        boxes_pixel = boxes * torch.Tensor([w, h, w, h])
        boxes_numpy = boxes_pixel.numpy()
        boxes_xyxy = np.column_stack([
            boxes_numpy[:, 0] - boxes_numpy[:, 2]/2,  # x1 = cx - w/2
            boxes_numpy[:, 1] - boxes_numpy[:, 3]/2,  # y1 = cy - h/2
            boxes_numpy[:, 0] + boxes_numpy[:, 2]/2,  # x2 = cx + w/2
            boxes_numpy[:, 1] + boxes_numpy[:, 3]/2   # y2 = cy + h/2
        ])
        # 3. SAM 2 STEP
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self.sam2_predictor.set_image(image_rgb)
        
        masks, scores, _ = self.sam2_predictor.predict(
            box=boxes_xyxy,
            multimask_output=False 
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        detections = sv.Detections(
            xyxy=boxes_xyxy,
            mask=masks.astype(bool),
            class_id=np.array([0] * len(boxes)), 
            confidence=logits.numpy()
        )
        detections.data['class_name'] = phrases
        
        return detections