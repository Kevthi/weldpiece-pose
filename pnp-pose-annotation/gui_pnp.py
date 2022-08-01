from gui_components import Sidebar, SidebarButton, ColBoxLayout, ColAnchorLayout
from gui_utils import ask_directory, ask_file, read_rgb
from gui_components import color_profile as cp
from renderer import render_scene
from render_utils import convert_cam_mat
import matplotlib.pyplot as plt


from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.core.window import Window

from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color

from kivy.uix.image import Image
from kivy.graphics.texture import Texture

from kivy.graphics.transformation import Matrix
import asyncio
from threading import Thread
import time

from image_component import DragZoomImage



import cv2
import numpy as np



class PnPSidebar(Sidebar):
    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        self.add_widget(SidebarButton(text="Add corr", height=50))
        self.add_widget(SidebarButton(height=50))
        self.add_widget(Widget())

class CameraImageDisplay(ColBoxLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, orientation='vertical', size_hint=(0.33,1.0))
        self.state_dict = state_dict
        img_path = state_dict["paths"]["selected_img"]
        rgb_img = read_rgb(img_path)
        self.state_dict["pnp"]["cam_img"] = rgb_img
        self.image_handler = DragZoomImage(rgb_img, cp.FILESELECT_BG)
        self.image_handler.bind_image_cb(self.on_click)
        self.add_widget(self.image_handler)


    def on_click(self, pixel_coord):
        print("cam img clicked", pixel_coord)
        self.state_dict["pnp"]["img_select"] = pixel_coord
        self.state_dict["functions"]["on_add_corr"]()

class ModelImageDisplay(ColBoxLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, orientation='vertical', size_hint=(0.33,1.0))
        self.state_dict = state_dict
        model_path = state_dict["paths"]["selected_model"]
        K = state_dict["scene"]["cam_K"]
        T_WC = state_dict["scene"]["T_wc"]
        orig_img_size = self.state_dict["scene"]["orig_img_size"]
        WIDTH = 4000
        K_new, img_size = convert_cam_mat(K, orig_img_size, WIDTH)
        print("K_new", K_new)
        print("img size", img_size)
        

        rgb_img, depth = render_scene(model_path, np.linalg.inv(T_WC), K_new, img_size)
        self.image_handler = DragZoomImage(rgb_img, cp.FILESELECT_BG)
        self.add_widget(self.image_handler)
        self.image_handler.bind_image_cb(self.on_click)

    def rerender_pnp(self):
        state_dict = self.state_dict
        model_path = state_dict["paths"]["selected_model"]
        K = state_dict["scene"]["cam_K"]
        T_WC = state_dict["scene"]["T_wc"]
        orig_img_size = self.state_dict["scene"]["orig_img_size"]
        WIDTH = 4000
        K_new, img_size = convert_cam_mat(K, orig_img_size, WIDTH)
        rgb_img, depth = render_scene(model_path, np.linalg.inv(T_WC), K_new, img_size)
        self.state_dict["pnp"]["rend_img"] = rgb_img
        self.state_dict["pnp"]["rend_depth"] = depth
        self.image_handler.set_texture(rgb_img)

    def on_click(self, pixel_coord):
        print("model click", pixel_coord)
        self.state_dict["pnp"]["rend_select"] = pixel_coord
        self.state_dict["functions"]["on_add_corr"]()

        


class OverlapImageDisplay(ColBoxLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, orientation='vertical', size_hint=(0.33,1.0))
        self.state_dict = state_dict
        img_path = state_dict["paths"]["selected_img"]
        self.rgb_img = read_rgb(img_path)
        self.image_handler = DragZoomImage(self.rgb_img, cp.FILESELECT_BG)
        self.add_widget(self.image_handler)

class ImageMainWin(ColBoxLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, orientation='horizontal', size_hint=(1.0,0.65))
        self.state_dict = state_dict
        self.state_dict["functions"]["on_add_corr"] = self.on_add_corr
        self.state_dict["functions"]["draw_corrs"] = self.draw_corrs
        cam_img_disp = CameraImageDisplay(state_dict)
        self.add_widget(cam_img_disp)
        self.add_widget(ColBoxLayout(cp.FS_DELIMITER, size_hint=(None,1.0), width=20))
        model_img_disp = ModelImageDisplay(state_dict)
        self.add_widget(model_img_disp)
        self.add_widget(ColBoxLayout(cp.FS_DELIMITER, size_hint=(None,1.0), width=20))
        overlap_img_display = OverlapImageDisplay(state_dict)
        self.add_widget(overlap_img_display)

    def on_add_corr(self):
        print("on add corr")
        left_select = self.state_dict["pnp"]["img_select"]
        right_select = self.state_dict["pnp"]["rend_select"]
        both_selected = (left_select is not None) and (right_select is not None)
        if both_selected:
            self.state_dict["pnp"]["img_select"] = None
            self.state_dict["pnp"]["rend_select"] = None
            self.state_dict["pnp"]["corresps"].append([left_select, right_select])
        self.draw_corrs()

    def draw_corrs(self):
        print("draw corrs")




class PnPGUI(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='horizontal')
        self.state_dict = state_dict
        self.add_widget(PnPSidebar(state_dict))
        self.vert_layout = BoxLayout(orientation='vertical', size_hint=(1.0, 1.0))
        self.add_widget(self.vert_layout)
        main_image_layout = ImageMainWin(state_dict)
        self.vert_layout.add_widget(main_image_layout)
        self.vert_layout.add_widget(ColBoxLayout(cp.FS_DELIMITER, size_hint=(1.0,None), height=20))

        bot_box_layot = ColBoxLayout(cp.FILESELECT_BG, orientation='horizontal', size_hint=(1.0,0.35))
        self.vert_layout.add_widget(bot_box_layot)









