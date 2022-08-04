import cv2
import numpy as np
import argparse

def command_line_parse_camera_num():
    parser = argparse.ArgumentParser()
    parser.add_argument("cam")
    args = parser.parse_args()
    return args

def crop_image_center(image, crop_size):
    crop_width = crop_size[0]
    crop_height = crop_size[1]
    image_height = image.shape[0]
    image_width = image.shape[1]
    amount_crop_width = image_width-crop_width
    crop_start_w = amount_crop_width//2
    crop_end_w = image_width-amount_crop_width//2
    amount_crop_height = image_height-crop_height
    crop_start_h = amount_crop_height//2
    crop_end_h = image_height-amount_crop_height//2
    cropped = image[crop_start_h:crop_end_h, crop_start_w:crop_end_w, :]
    return cropped



class WebcamHandler:
    def __init__(self, webcam_device_num, camera_config_dict, video_capture_flag=cv2.CAP_ANY):
        self.crop_size = None
        self.cam_config = camera_config_dict
        self.cam = None
        cam = cv2.VideoCapture(webcam_device_num, video_capture_flag)
        self.cam = cam
        if not cam.isOpened():
            raise IOError("The webcam could not be opened, try changing video_capture_flag")
        self.set_from_config(camera_config_dict)

    def set_from_config(self, cam_config):
        cam = self.cam
        self.cam_config = cam_config
        capture_res = cam_config["capture_resolution"]
        #capture_size = cam_config["valid_frame_sizes"][capture_res]
        print("set from config", capture_res)
        self.set_capture_frame_size(capture_res)
        self.cam = cam
        cam_focus = self.cam_config['focus_distance']
        use_autofocus = cam_config['use_autofocus']
        if use_autofocus:
            self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        else: 
            self.set_manual_focus(cam_focus)

    def set_crop_size(self, crop_size):
        self.cam_config["crop_size"] = crop_size

    def get_crop_size(self):
        return self.cam_config["crop_size"]

    def set_resize_to(self, resize_to):
        self.cam_config["resize_to"] = resize_to

    def get_resize_to(self):
        return self.cam_config["resize_to"]

    def get_config(self):
        return self.cam_config

    def get_camera_image(self):
        ret, frame = self.cam.read()
        if not ret:
            return ret, frame

        crop_size = self.cam_config["crop_size"]
        if(crop_size is not None):
            frame = crop_image_center(frame, crop_size)

        resize_to = self.cam_config["resize_to"]
        if(resize_to is not None):
            frame = cv2.resize(frame, resize_to)
        return ret, frame

    def set_capture_frame_size(self, capture_size):
        self.cam_config["capture_resolution"] = capture_size
        capture_size = self.cam_config["valid_frame_sizes"][capture_size]
        width = capture_size[0]
        height = capture_size[1]
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)







    def set_manual_focus(self, focus):
        assert focus >= 0 and focus <= 51
        assert isinstance(focus, int)
        self.cam_config["focus_distance"] = focus
        if(focus is not None):
            OPENCV_FOCUS_INCREMENT = 5
            opencv_focus = OPENCV_FOCUS_INCREMENT*focus
            self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            self.cam.set(28, opencv_focus)

    def get_focus(self):
        return self.cam.get(28)//5


    def __del__(self):
        if(self.cam is not None):
            self.cam.release()

    def get_video_capture(self):
        return self.cam
    
    def show_video_capture(self):
        cam = self.cam
        idx = 0
        main_win = "Main"
        cv2.namedWindow(main_win)

        while True:
            ret, frame = cam.read()
            print(frame.shape)
            cv2.imshow(main_win, frame)
            c = cv2.waitKey(1)
            if(c == 27 or c == ord('q') or c==ord('Q')): # 27 = Escape key
                break
            if c == ord('s'):
                print("Saving image")
                save_path = os.path.join(SAVE_DIR, "img"+format(idx, "03d")+".png")
                cv2.imwrite(save_path, frame)
                idx += 1
        cv2.destroyAllWindows()

    

if __name__ == '__main__':
    pass
    args = command_line_parse_camera_num()
    cam = int(args.cam)
    cam_conf = {
        "valid_frame_sizes": {
            "1920x1080": (1920,1080),
            "1280x720": (1280,720),
            "640x480": (640,480),
            "320x240": (320,240),
            "160x120": (160,120)
        },
        "capture_resolution": "1920x1080",
        "focus_distance": 0,
        "use_autofocus": False,
        "crop_size": (100,100),
        "resize_to": (1000,1000),
    }
    print("Opening camera", cam)
    webcam_handler = WebcamHandler(cam, cam_conf)
    webcam_handler.show_video_capture()

