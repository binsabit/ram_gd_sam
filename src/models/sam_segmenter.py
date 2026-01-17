import cv2
import torch
import numpy as np
from PIL import Image
from groundingdino.util.inference import load_model, load_image, predict
import groundingdino.datasets.transforms as T
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


class Segmentor:

    def __init__(gd_path='',sam_path='',device=None ):

        gd_model = load_model(GD_CONFIG, GD_CHECKPOINT, device=device)

        print("Loading SAM 2...")
        sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)     