from util.config_loader import load_config
from inputs.intel_realsense import RealsenseCamera
from inputs.base_camera import Camera
import cv2
import numpy as np
from models.ram_tagger import RAMImageTagger

def run_pipeline():
    cfg = load_config("config/settings.yaml")
    
    
    device = cfg.models.device


    camera = Camera()

    if cfg.input_source == "realsense":
        camera = RealsenseCamera(
            width=cfg.realsense.width,
            height=cfg.realsense.height,
            fps=cfg.realsense.fps)
    else:
        raise("Unknown input source")

    
    camera.start()

    ram = RAMImageTagger(cfg.models.ram_weights, device=cfg.models.device)


    while True:
        frame = camera.wait_for_frames()

        depth_channel = frame.get_depth_channel()
        color_channel = frame.get_color_channel()

        depth_image = np.asanyarray(depth_channel.get_data())
        color_image = np.asanyarray(color_channel.get_data())
        
        depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,
                                                         alpha=0.5), cv2.COLORMAP_JET)

        cv2.imshow('depth', depth_cm)
        cv2.imshow('rgb', color_image) 

        
        tags = ram.recognize(color_image)

        print(tags)
        
        if cv2.waitKey(1) == ord('q'):
            break

            


    print(f"Starting {cfg.project_name} in debug mode: {cfg.debug_mode}")
    pass

if __name__ == "__main__":
    run_pipeline()