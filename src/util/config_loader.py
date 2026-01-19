import yaml 
from dataclasses import dataclass
from typing import List, Tuple

# 1. Define the structure of your data
@dataclass
class RealsenseConfig:
    width: float
    height: float
    fps: int

@dataclass
class VideoConfig:
    path: str
    loop: bool

@dataclass
class Model:
    weights: str
    config: str

@dataclass
class ModelConfig:
    sam_checkpoint: Model
    grounding_dino: Model
    recognize_anything: Model
    device: str

@dataclass
class AppConfig:
    project_name: str
    debug_mode: bool
    input_source: str
    realsense: RealsenseConfig
    video: VideoConfig
    models: ModelConfig

# 2. The Loading Logic
def load_config(config_path: str) -> AppConfig:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    # Convert the raw dictionary into our strict DataClasses
    return AppConfig(
        project_name=data['project']['name'],
        debug_mode=data['project']['debug_mode'],
        input_source=data['input']['source'],
        realsense=RealsenseConfig(
            width=data['realsense']['width'],
            height=data['realsense']['height'],
            fps=data['realsense']['fps'],
        ),
        video=VideoConfig(
            path=data['video_file']['path'],
            loop=data['video_file']['loop']
        ),
        models=ModelConfig(
            sam_checkpoint=Model(
                weights=data['models']['sam_checkpoint']['weights'],
                config=data['models']['sam_checkpoint']['config']
                ),
            grounding_dino=Model(
                weights=data['models']['grounding_dino']['weights'],
                config=data['models']['grounding_dino']['config']
                ),
            recognize_anything=Model(
                weights=data['models']['recognize_anything']['weights'],
                config=data['models']['recognize_anything']['config']
                ),
            device=data['models']['device']
        )
    )