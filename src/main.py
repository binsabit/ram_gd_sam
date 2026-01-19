from util.config_loader import load_config
from inputs.intel_realsense import RealsenseCamera
from inputs.base_camera import Camera
import cv2
import numpy as np
from models.sam_segmenter import Segmentor
import supervision as sv
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

    ram = RAMImageTagger(cfg.models.recognize_anything.weights, device=cfg.models.device)


    segmentor = Segmentor(
        gd_config = cfg.models.grounding_dino.config,
        gd_ckpt = cfg.models.grounding_dino.weights,
        sam2_config  = cfg.models.sam_checkpoint.config,
        sam2_ckpt = cfg.models.sam_checkpoint.weights,
        device = "cuda"
    )


    filename = 'annotated_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    fps = 30.0

    frame_size = (640,480)


    out = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=fps,frameSize=frame_size)

    try:
        while True:
            frame = camera.wait_for_frames()
            color_image = np.asanyarray(frame.get_color_channel().get_data())
            
            # 0. tag the image
            recognized_objects = ram.recognize(image=color_image) 

            if len(recognized_objects) == 0:
                continue
            # 1. Process Segmentation
            results = segmentor.process_image(color_image, text_prompt=", ".join(recognized_objects))
            
            # 2. Annotate
            if len(results) > 0:
                mask_annotator = sv.MaskAnnotator()
                box_annotator = sv.BoxAnnotator()
                label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
                
                annotated_img = mask_annotator.annotate(scene=color_image.copy(), detections=results)
                annotated_img = box_annotator.annotate(scene=annotated_img, detections=results)
                
                if 'class_name' in results.data:
                    labels = [f"{name} {conf:.2f}" for name, conf in zip(results.data['class_name'], results.confidence)]
                    annotated_img = label_annotator.annotate(scene=annotated_img, detections=results, labels=labels)
                
                display_frame = annotated_img
            else:
                # IMPORTANT: Still write the frame even if no one is detected
                display_frame = color_image

            # 3. Resize and Write
            # Ensure display_frame is BGR (OpenCV default)
            resized_image = cv2.resize(display_frame, frame_size)
            out.write(resized_image)

            # 4. Display
            cv2.imshow('annotated', resized_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # This MUST run to make the video playable
        print("Finalizing video file...")
        out.release()
        camera.stop() # Good practice to stop the RealSense stream
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    run_pipeline()