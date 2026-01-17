import pyrealsense2 as rs
from inputs.base_camera import Camera, Frame


class RealsenseCameraFrame(Frame):
    frame: any

    def __init__(self, frame):
        self.frame = frame


    def get_depth_channel(self):
        return self.frame.get_depth_frame()

    def get_color_channel(self):
        return self.frame.get_color_frame()


class RealsenseCamera(Camera):
    pipe: any
    cfg: any
    profile: any


    def __init__(self,width: float = 640, height: float = 480, fps: int = 30):

        pipe = rs.pipeline()
    
        cfg = rs.config()
    
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)


        self.cfg = cfg
        self.pipe = pipe
        
    def start(self):
        self.profile = self.pipe.start(self.cfg)

    def wait_for_frames(self):
        return RealsenseCameraFrame(frame =self.pipe.wait_for_frames())
    
    def stop(self):
        self.pipe.stop()


