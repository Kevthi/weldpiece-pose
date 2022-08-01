from gui_components import Sidebar, SidebarButton, TextureImage, ColAnchorLayout, ColBoxLayout
from gui_utils import ask_directory, ask_file, read_rgb
from render_utils import get_optimal_camera_pose, convert_cam_mat
from renderer import PersistentRenderer
import time
import spatialmath as sm
from debug import *

from gui_components import color_profile as cp

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color


class InitPoseSidebar(Sidebar):
    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        self.print_state_dict_btn = SidebarButton(text="Print state dict")
        self.print_state_dict_btn.bind(on_press=self.print_state_dict)
        self.add_widget(self.print_state_dict_btn)
        self.add_widget(PoseButtonHandler(state_dict))



        self.add_widget(Widget())

    def print_state_dict(self, instance):
        print("\n\n ### State dict ###")
        nice_print_dict(self.state_dict)

class PoseButtonHandler(BoxLayout):
    def __init__(self,state_dict):
        super().__init__(orientation='vertical',size_hint=(1.0,None), height=180)
        self.state_dict = state_dict
        z_incr_decr = IncrementDecrementButtons(state_dict, "Decrement Z rot", "Increment Z rot", self.decr_z, self.incr_z)
        self.add_widget(z_incr_decr)
        x_incr_decr = IncrementDecrementButtons(state_dict, "Decrement X rot", "Increment X rot", self.decr_x, self.incr_x)
        self.add_widget(x_incr_decr)
        y_incr_decr = IncrementDecrementButtons(state_dict, "Decrement Y rot", "Increment Y rot", self.decr_y, self.incr_y)
        self.add_widget(y_incr_decr)
        flip_z_btn = SidebarButton(text="Flip Z")
        flip_z_btn.bind(on_press=self.flip_z)
        self.add_widget(flip_z_btn)
        flip_y_btn = SidebarButton(text="Flip Y")
        flip_y_btn.bind(on_press=self.flip_y)
        self.add_widget(flip_y_btn)
        flip_x_btn = SidebarButton(text="Flip X")
        flip_x_btn.bind(on_press=self.flip_x)
        self.add_widget(flip_x_btn)


    def decr_z(self,instance):
        rot = sm.SE3.Rz(-10, unit='deg').data[0]
        T_wc = self.state_dict["scene"]["T_wc"] 
        T_wc_new = rot@T_wc
        self.state_dict["scene"]["T_wc"] = T_wc_new
        self.state_dict["functions"]["rerender"]()


    def incr_z(self,instance):
        rot = sm.SE3.Rz(10, unit='deg').data[0]
        T_wc = self.state_dict["scene"]["T_wc"] 
        T_wc_new = rot@T_wc
        self.state_dict["scene"]["T_wc"] = T_wc_new
        self.state_dict["functions"]["rerender"]()

    def decr_x(self,instance):
        rot = sm.SE3.Rx(-10, unit='deg').data[0]
        T_wc = self.state_dict["scene"]["T_wc"] 
        T_wc_new = rot@T_wc
        self.state_dict["scene"]["T_wc"] = T_wc_new
        self.state_dict["functions"]["rerender"]()

    def incr_x(self,instance):
        rot = sm.SE3.Rx(10, unit='deg').data[0]
        T_wc = self.state_dict["scene"]["T_wc"] 
        T_wc_new = rot@T_wc
        self.state_dict["scene"]["T_wc"] = T_wc_new
        self.state_dict["functions"]["rerender"]()

    def decr_y(self,instance):
        rot = sm.SE3.Ry(-10, unit='deg').data[0]
        T_wc = self.state_dict["scene"]["T_wc"] 
        T_wc_new = rot@T_wc
        self.state_dict["scene"]["T_wc"] = T_wc_new
        self.state_dict["functions"]["rerender"]()

    def incr_y(self,instance):
        rot = sm.SE3.Ry(10, unit='deg').data[0]
        T_wc = self.state_dict["scene"]["T_wc"] 
        T_wc_new = rot@T_wc
        self.state_dict["scene"]["T_wc"] = T_wc_new
        self.state_dict["functions"]["rerender"]()

    def flip_x(self,instance):
        rot = sm.SE3.Rx(180, unit='deg').data[0]
        T_wc = self.state_dict["scene"]["T_wc"] 
        T_wc_new = rot@T_wc
        self.state_dict["scene"]["T_wc"] = T_wc_new
        self.state_dict["functions"]["rerender"]()

    def flip_y(self,instance):
        rot = sm.SE3.Ry(180, unit='deg').data[0]
        T_wc = self.state_dict["scene"]["T_wc"] 
        T_wc_new = rot@T_wc
        self.state_dict["scene"]["T_wc"] = T_wc_new
        self.state_dict["functions"]["rerender"]()

    def flip_z(self,instance):
        rot = sm.SE3.Rz(180, unit='deg').data[0]
        T_wc = self.state_dict["scene"]["T_wc"] 
        T_wc_new = rot@T_wc
        self.state_dict["scene"]["T_wc"] = T_wc_new
        self.state_dict["functions"]["rerender"]()

class CameraImageWindow(ColAnchorLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, anchor_x="center", anchor_y="center", size_hint=(0.5,1.0))
        self.state_dict = state_dict
        img_path = state_dict["paths"]["selected_img"]
        img = read_rgb(img_path)
        self.add_widget(TextureImage(img))

class RenderImageWindow(ColAnchorLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, anchor_x="center", anchor_y="center", size_hint=(0.5,1.0))
        self.state_dict = state_dict
        self.renderer = None
        self.init_renderer()
        T_wc = self.state_dict["scene"]["T_wc"]
        img, depth = self.renderer.render(T_wc)
        self.texture_img = TextureImage(img)
        self.add_widget(self.texture_img)
        self.state_dict["functions"]["rerender"] = self.rerender





    def init_renderer(self):
        model_path = self.state_dict["paths"]["selected_model"]
        K = self.state_dict["scene"]["cam_K"]
        orig_img_size = self.state_dict["scene"]["orig_img_size"]
        NEW_WIDTH = 2000
        K,img_size = convert_cam_mat(K, orig_img_size, NEW_WIDTH)

        print("K",K)
        T_wc = self.state_dict["scene"]["T_wc"]
        self.renderer = PersistentRenderer(model_path,K, T_wc, img_size)

    def rerender(self):
        T_wc = self.state_dict["scene"]["T_wc"]
        img, depth = self.renderer.render(T_wc)
        self.texture_img.update_texture(img)



class IncrementDecrementButtons(BoxLayout):
    def __init__(self, state_dict, text_dec, text_incr, cb_dec, cb_inc):
        super().__init__(orientation='horizontal', height=30, size_hint=(1.0,None))
        decr_btn = SidebarButton(text=text_dec)
        decr_btn.bind(on_press=cb_dec)
        self.add_widget(decr_btn)
        incr_btn = SidebarButton(text=text_incr)
        incr_btn.bind(on_press=cb_inc)
        self.add_widget(incr_btn)



class InitPoseGUI(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='horizontal')
        self.state_dict = state_dict
        self.init_first_view_pose()
        self.add_widget(InitPoseSidebar(state_dict))
        self.add_widget(CameraImageWindow(state_dict))
        self.add_widget(ColBoxLayout(cp.FS_DELIMITER, size_hint=(None,1.0), width=20))
        self.add_widget(RenderImageWindow(state_dict))


    def init_first_view_pose(self):
        t_wc_saved = self.state_dict["scene"]["T_wc"]
        mesh_path = self.state_dict["paths"]["selected_model"]
        if t_wc_saved is None:
            self.state_dict["scene"]["T_wc"] = get_optimal_camera_pose(mesh_path, 3.0).data[0]











