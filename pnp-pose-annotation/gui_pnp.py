from gui_components import Sidebar, SidebarButton, ColBoxLayout, ColAnchorLayout, SidebarHeader 
from gui_utils import ask_directory, ask_file, read_rgb, draw_corresps_both, get_color_list, split_corresps, blend_imgs
from gui_components import color_profile as cp
from renderer import render_scene
from render_utils import convert_cam_mat
import matplotlib.pyplot as plt
from pnp_handler import project_to_3d, solve_pnp_ransac, transform_points, convert_indices_to_pix_coord


from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.core.window import Window
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.uix.scrollview import  ScrollView
from kivy.uix.gridlayout import GridLayout

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
import rhovee



import cv2
import numpy as np
import os



class PnPSidebar(Sidebar):
    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        self.slider = None
        self.slider_label = None
        self.add_marker_slidebar()
        self.add_blend_slidebar()
        self.add_reproj_error_slidebar()
        self.corr_display = AllCorrDisplay(state_dict)
        self.add_widget(self.corr_display)
        self.add_widget(SavePoseBtn(state_dict))
        self.add_widget(Widget())


    def add_marker_slidebar(self):
        box_layout = BoxLayout(size_hint=(1.0,None), height=30)
        cur_mark_sz = self.state_dict["pnp"]["marker_size"]
        slider = Slider(min=1,max=12, step=1, value=cur_mark_sz, value_track_color=cp.PNP_SLIDER_BG, value_track_width=10, size_hint=(0.6,1.0))
        self.marker_slider_label = Label(text=str(slider.value), size_hint=(0.1,1.0), color=cp.BLACK_TEXT)
        box_layout.add_widget(Label(text="Marker size", size_hint=(0.3,1.0), color=cp.BLACK_TEXT))
        box_layout.add_widget(slider)
        box_layout.add_widget(self.marker_slider_label)
        self.add_widget(box_layout)
        slider.bind(value=self.update_label)

    def add_blend_slidebar(self):
        box_layout = BoxLayout(size_hint=(1.0,None), height=30)
        blend_alpha = self.state_dict["pnp"]["blend_alpha"]
        print("blend alpha", blend_alpha)
        bl_slider = Slider(min=0,max=100, step=10, value=blend_alpha*100, value_track_color=cp.PNP_SLIDER_BG, value_track_width=10, size_hint=(0.6,1.0))
        self.blend_slider_label = Label(text=str(blend_alpha), size_hint=(0.1,1.0), color=cp.BLACK_TEXT)
        box_layout.add_widget(Label(text="Blend alpha", size_hint=(0.3,1.0), color=cp.BLACK_TEXT))
        box_layout.add_widget(bl_slider)
        box_layout.add_widget(self.blend_slider_label)
        self.add_widget(box_layout)
        bl_slider.bind(value=self.update_blend_alpha)

    def add_reproj_error_slidebar(self):
        box_layout = BoxLayout(size_hint=(1.0,None), height=30)
        reproj_error = self.state_dict["pnp"]["reproj_error"]
        bl_slider = Slider(min=0,max=40, step=1, value=reproj_error, value_track_color=cp.PNP_SLIDER_BG, value_track_width=10, size_hint=(0.6,1.0))
        self.reproj_slider_label = Label(text=str(reproj_error), size_hint=(0.1,1.0), color=cp.BLACK_TEXT)
        box_layout.add_widget(Label(text="Outlier treshold", size_hint=(0.3,1.0), color=cp.BLACK_TEXT))
        box_layout.add_widget(bl_slider)
        box_layout.add_widget(self.reproj_slider_label)
        self.add_widget(box_layout)
        bl_slider.bind(value=self.update_reproj_error)

    def update_reproj_error(self, instance, value):
        self.reproj_slider_label.text = str(value)
        self.state_dict["pnp"]["reproj_error"] = float(value)
        self.state_dict["functions"]["update_pnp"]()


    def update_blend_alpha(self, instance, value):
        self.blend_slider_label.text = str(value/100.0)
        self.state_dict["pnp"]["blend_alpha"] = value/100.0
        self.state_dict["functions"]["show_overlap_pnp"]()

    def update_label(self, instance, value):
        value = int(instance.value)
        self.marker_slider_label.text = str(value)
        self.state_dict["pnp"]["marker_size"] = value
        self.state_dict["functions"]["draw_corrs"]()

class SavePoseBtn(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='vertical', size_hint=(1.0,None), height=100)
        self.state_dict = state_dict
        header = SidebarHeader(text="Save current pose", font_size=20)
        self.add_widget(header)
        self.state_dict["functions"]["render_save_pose_btn"] = self.render_save_pose_btn

    def render_save_pose_btn(self):
        corresps = self.state_dict["pnp"]["corresps"]
        num_corrs = len(corresps)
        self.clear_widgets()
        header = SidebarHeader(text="Save current pose", font_size=20)
        self.add_widget(header)
        if (num_corrs < 5):
            self.add_widget(Button(text=f'Need {5-num_corrs} more correspondences', size_hint=(1.0,None), height=50))
        else:
            save_pose_btn = Button(text=f'Save', size_hint=(1.0,None), height=50, background_color=cp.PNP_SAVE_POSE_BTN)
            self.add_widget(save_pose_btn)
            save_pose_btn.bind(on_press=self.on_save_pose)

    def on_save_pose(self, btn_instance=None):
        img_name = os.path.basename(self.state_dict["paths"]["selected_img"])
        T_WC_pnp = self.state_dict["pnp"]["T_WC_pnp"]
        self.state_dict["pose_dict"][img_name]["T_CO"] = rhovee.SE3.inv(T_WC_pnp)
        self.state_dict["pose_dict"][img_name]["pose_set_with_pnp"] = True
        print(f'Saving pose to {img_name}')










        

class SingleCorrDisplay(BoxLayout):
    def __init__(self, state_dict, color, idx, corr_tup):
        super().__init__(orientation='horizontal', size_hint=(1.0, None), height=46, padding=[20,2,20,2])
        self.state_dict = state_dict
        self.idx = idx
        self.corr_tup = corr_tup
        self.anchor_label = AnchorLayout(size_hint=(0.7,None), height=30, anchor_x='center', anchor_y='center')

        self.inner_box = ColBoxLayout(cp.PNP_SIDEBAR_CORR, size_hint=(0.9, None), height=30)

        label_text = str(corr_tup[0]) + "  " +str(corr_tup[1])
        self.label = SidebarHeader(text=label_text, font_size=16)
        self.anchor_label.add_widget(self.label)

        remove_btn = Button(text='X', size_hint=(None,None), width=30, height=30)
        remove_btn.bind(on_press=self.on_remove)


        self.inner_box.add_widget(ColBoxLayout(color, size_hint=(None,None), height=30, width=30))
        self.inner_box.add_widget(self.anchor_label)
        self.inner_box.add_widget(remove_btn)
        self.add_widget(self.inner_box)

    def on_remove(self, instance):
        self.state_dict["pnp"]["corresps"].pop(self.idx)
        self.state_dict["functions"]["draw_corrs"]()



class AllCorrDisplay(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='vertical', size_hint=(1.0,None), height=500, padding=[0,5,0,0])
        header = SidebarHeader(text="Correspondonces", font_size=20)
        self.add_widget(header)

        self.inner_box = ColBoxLayout(cp.PNP_CORR_WIN_BG, orientation='vertical')
        self.add_widget(self.inner_box)

        self.scroll_view = ScrollView(do_scroll_y=True, bar_width=20)
        self.inner_box.add_widget(self.scroll_view)
        self.state_dict = state_dict
        self.state_dict["functions"]["update_sidebar_corrs"] = self.update_sidebar_corrs
        self.grid = GridLayout(cols=1, size_hint_y=None)
        self.grid.bind(minimum_height=self.grid.setter('height'))
        self.scroll_view.add_widget(self.grid)
        self.update_sidebar_corrs()

    def update_sidebar_corrs(self):
        self.grid.clear_widgets()
        corresps = self.state_dict["pnp"]["corresps"]
        color_list = get_color_list(len(corresps))
        for idx,color in enumerate(color_list):
            self.grid.add_widget(SingleCorrDisplay(self.state_dict, (color[0]/255,color[1]/255, color[2]/255), idx, corresps[idx]))
            #self.add_widget(Widget())








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

    def set_texture(self, rgb_img):
        self.image_handler.set_texture(rgb_img)

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

        

        rgb_img, depth = render_scene(model_path, rhovee.SE3.inv(T_WC), K_new, img_size)
        self.state_dict["pnp"]["rend_img"] = rgb_img
        self.state_dict["pnp"]["rend_depth"] = depth
        self.state_dict["pnp"]["rend_K"] = K_new
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
        rend_depth = self.state_dict["pnp"]["rend_depth"]
        if (rend_depth[pixel_coord[0], pixel_coord[1]]>0):
            self.state_dict["pnp"]["rend_select"] = pixel_coord
            self.state_dict["functions"]["on_add_corr"]()

    def set_texture(self, rgb_img):
        self.image_handler.set_texture(rgb_img)


        



class ImageMainWin(ColBoxLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, orientation='horizontal', size_hint=(1.0,0.65))
        self.state_dict = state_dict
        self.state_dict["functions"]["on_add_corr"] = self.on_add_corr
        self.state_dict["functions"]["draw_corrs"] = self.draw_corrs
        self.cam_img_disp = CameraImageDisplay(state_dict)
        self.add_widget(self.cam_img_disp)
        self.add_widget(ColBoxLayout(cp.FS_DELIMITER, size_hint=(None,1.0), width=20))
        self.model_img_disp = ModelImageDisplay(state_dict)
        self.add_widget(self.model_img_disp)
        self.add_widget(ColBoxLayout(cp.FS_DELIMITER, size_hint=(None,1.0), width=20))
        overlap_img_display = OverlapImageDisplay(state_dict)
        self.add_widget(overlap_img_display)
        self.draw_corrs()

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

        cam_img = self.state_dict["pnp"]["cam_img"]
        rend_img = self.state_dict["pnp"]["rend_img"]
        img_select = self.state_dict["pnp"]["img_select"]
        rend_select = self.state_dict["pnp"]["rend_select"]
        corresps = self.state_dict["pnp"]["corresps"]
        marker_size = self.state_dict["pnp"]["marker_size"]
        cam_img_mark, rend_img_mark = draw_corresps_both(cam_img, rend_img, corresps, img_select, rend_select, marker_size)
        self.cam_img_disp.set_texture(cam_img_mark)
        self.model_img_disp.set_texture(rend_img_mark)
        self.state_dict["functions"]["update_sidebar_corrs"]()
        self.state_dict["functions"]["update_pnp"]()

        

class OverlapImageDisplay(ColBoxLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, orientation='vertical', size_hint=(0.33,1.0))
        self.state_dict = state_dict
        self.state_dict["functions"]["update_pnp"] = self.update_pnp
        self.state_dict["functions"]["show_overlap_pnp"] = self.show_overlap_pnp
        img_path = state_dict["paths"]["selected_img"]
        self.rgb_img = read_rgb(img_path)
        self.image_handler = None

    def update_pnp(self):
        corresps = self.state_dict["pnp"]["corresps"]
        num_corrs = len(corresps)
        self.state_dict["functions"]["render_save_pose_btn"]()
        if num_corrs < 5:
            if self.image_handler is not None:
                self.remove_widget(self.image_handler)
                self.image_handler = None
            return
        else:
            if self.image_handler is None:
                self.image_handler = DragZoomImage(self.rgb_img, cp.FILESELECT_BG)
                self.add_widget(self.image_handler)

        rend_K = self.state_dict["pnp"]["rend_K"]
        rend_depth = self.state_dict["pnp"]["rend_depth"]
        img_corrs, rend_corrs = split_corresps(corresps)
        cam_K = self.state_dict["scene"]["cam_K"]
        T_WC = self.state_dict["scene"]["T_wc"]
        T_CW_guess = np.linalg.inv(T_WC)

        rend_pix_coords = convert_indices_to_pix_coord(rend_corrs)
        img_pix_coords = convert_indices_to_pix_coord(img_corrs)

        points_C = project_to_3d(rend_pix_coords, rend_depth, rend_K)
        points_W = transform_points(points_C.T, T_WC)

        outlier_tresh = self.state_dict["pnp"]["reproj_error"]
        T_CW, inliers = solve_pnp_ransac(points_W.T, img_pix_coords, cam_K, outlier_tresh, T_CW_guess)
        print("inliers", inliers)
        self.state_dict["pnp"]["T_WC_pnp"] = np.linalg.inv(T_CW)
        self.render_pnp_pose()

    def render_pnp_pose(self):
        T_CW = np.linalg.inv(self.state_dict["pnp"]["T_WC_pnp"])
        obj_path = self.state_dict["paths"]["selected_model"]
        cam_K = self.state_dict["scene"]["cam_K"]
        print("cam K", cam_K)
        img_size = self.state_dict["pnp"]["cam_img"].shape[:2]
        rgb, depth = render_scene(obj_path, T_CW, cam_K, img_size)
        self.state_dict["pnp"]["rend_img_pnp"] = rgb
        self.state_dict["pnp"]["rend_depth_pnp"] = depth
        self.show_overlap_pnp()


    def show_overlap_pnp(self):
        corresps = self.state_dict["pnp"]["corresps"]
        num_corrs = len(corresps)
        if num_corrs < 5:
            return
        rend_pnp = self.state_dict["pnp"]["rend_img_pnp"] 
        depth_pnp = self.state_dict["pnp"]["rend_depth_pnp"]
        mask = np.where(depth_pnp>0, 1, 0)
        cam_img = self.state_dict["pnp"]["cam_img"]
        blend_alpha = self.state_dict["pnp"]["blend_alpha"]
        overlap = blend_imgs(cam_img, rend_pnp, mask, blend_alpha)
        self.image_handler.set_texture(overlap)









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













