from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from gui_components import Sidebar, ColBoxLayout, TextureImage, ColScrollView, SidebarHeader
from gui_components import color_profile as cp
from gui_utils import get_image_paths_from_dir, read_rgb
from aruco_graphopt import draw_markers
from cv2 import aruco as aruco





class ArucoGraphoptSidebar(Sidebar):
    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        self.add_widget(ImgFilesScroll(state_dict))
        self.add_widget(Widget())

class ImgFilesScroll(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='vertical', size_hint=(1.0,None), height=500, padding=[0,5,0,0])
        self.state_dict = state_dict
        self.add_widget(SidebarHeader(text="Images", font_size=20))



class ArucoGraphoptWorkspace(ColBoxLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, orientation="horizontal")
        self.state_dict = state_dict
        self.add_widget(ArucoDetectDisplay(state_dict))

class ArucoDetectDisplay(TextureImage):
    def __init__(self, state_dict):
        self.state_dict = state_dict
        self.state_dict["functions"]["update_aruco_detect_display"] = self.update_aruco_detect_display
        rgb_img = self.get_aruco_display_img()
        super().__init__(rgb_img)

    def get_aruco_display_img(self):
        img_dir = self.state_dict["paths"]["image_dir"]
        image_paths = get_image_paths_from_dir(img_dir)
        image_paths.sort()
        selected_img_idx = self.state_dict["aruco"]["selected_img_idx"]
        selected_img = image_paths[selected_img_idx]
        K = self.state_dict["scene"]["cam_K"]
        rgb_img = read_rgb(selected_img)
        rgb_img = draw_markers(rgb_img, K,"DICT_APRILTAG_16h5")
        return rgb_img

    def update_aruco_detect_display(self):
        rgb_img = self.get_aruco_display_img()
        self.update_texture(rgb_img)












class ArucoGraphoptGUI(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='horizontal')
        self.state_dict = state_dict
        self.add_widget(ArucoGraphoptSidebar(state_dict))
        self.add_widget(ArucoGraphoptWorkspace(state_dict))

