from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from gui_components import Sidebar, ColBoxLayout, TextureImage, ColScrollView, SidebarHeader
from gui_components import color_profile as cp
from image_component import DragZoomImage
from gui_utils import get_image_paths_from_dir, read_rgb, blend_imgs
from aruco_graphopt import aruko_optimize_handler
from charuco_board_utils import draw_markers_board
from gui_utils import get_image_paths_from_dir, merge_dict
from renderer import render_scene
import os
from cv2 import aruco as aruco
import numpy as np
from debug import nice_print_dict
from gan_remove_aruko.remove_aruko_board_gan import remove_aruco_boards_handler



class ArucoGraphoptSidebar(Sidebar):
    def __init__(self, state_dict):
        super().__init__()
        self.state_dict = state_dict
        self.add_widget(ImgFilesScroll(state_dict))
        self.add_widget(SolveGraphoptBtn(state_dict))
        self.add_blend_slidebar()
        self.add_widget(Widget())

    def add_blend_slidebar(self):
        box_layout = BoxLayout(size_hint=(1.0,None), height=30)
        blend_alpha = self.state_dict["aruco"]["blend_alpha"]
        print("blend alpha", blend_alpha)
        bl_slider = Slider(min=0,max=100, step=10, value=blend_alpha*100, value_track_color=cp.PNP_SLIDER_BG, value_track_width=10, size_hint=(0.6,1.0))
        self.blend_slider_label = Label(text=str(blend_alpha), size_hint=(0.1,1.0), color=cp.BLACK_TEXT)
        box_layout.add_widget(Label(text="Blend alpha", size_hint=(0.3,1.0), color=cp.BLACK_TEXT))
        box_layout.add_widget(bl_slider)
        box_layout.add_widget(self.blend_slider_label)
        self.add_widget(box_layout)
        bl_slider.bind(value=self.update_blend_alpha)

    def update_blend_alpha(self, instance, value):
        self.blend_slider_label.text = str(value/100.0)
        self.state_dict["aruco"]["blend_alpha"] = value/100.0
        # call funcs here
        self.state_dict["functions"]["show_overlap_graphopt"]()




class SolveGraphoptBtn(Button):
    def __init__(self, state_dict):
        super().__init__(text="Solve graph opt", size_hint=(1.0,None), height=50)
        self.state_dict = state_dict
        self.bind(on_press=self.on_solve_graph_opt)

    def on_solve_graph_opt(self, btn_instance=None):
        self.state_dict["aruco"]["graphopt_is_solved"] = True
        img_dir = self.state_dict["paths"]["image_dir"]
        img_paths = get_image_paths_from_dir(img_dir)
        K = self.state_dict["scene"]["cam_K"]
        ar_dict_str = "DICT_APRILTAG_16H5"
        ar_sq_size = 66.0*1e-3
        pose_dict = self.state_dict["pose_dict"]
        opt_pose_dict = aruko_optimize_handler(img_paths, K, ar_dict_str, ar_sq_size, pose_dict)
        nice_print_dict(opt_pose_dict)
        
        for key in self.state_dict["pose_dict"]:
            self.state_dict["pose_dict"][key]["T_CO_opt"] = opt_pose_dict[key]["T_CO_opt"]
        print(opt_pose_dict)
        self.state_dict["functions"]["render_aruco_graphopt_display_img"]()




class ImgFilesScroll(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='vertical', size_hint=(1.0,None), height=500, padding=[15,5,15,0])
        self.state_dict = state_dict
        self.state_dict["functions"]["render_aruco_img_select"] = self.render_aruco_img_select
        self.add_widget(SidebarHeader(text="Images", font_size=20))

        self.inner_box = ColBoxLayout(cp.PNP_CORR_WIN_BG, orientation='vertical')
        self.add_widget(self.inner_box)

        self.scroll_view = ScrollView(do_scroll_y=True, bar_width=5)
        self.inner_box.add_widget(self.scroll_view)
        self.grid = GridLayout(cols=1, size_hint_y=None)
        self.grid.bind(minimum_height=self.grid.setter('height'))
        self.scroll_view.add_widget(self.grid)

        self.render_aruco_img_select()


    def render_aruco_img_select(self):
        self.grid.clear_widgets()
        img_dir = self.state_dict["paths"]["image_dir"]
        image_paths = get_image_paths_from_dir(img_dir)
        image_paths.sort()
        selected_img_idx = self.state_dict["aruco"]["selected_img_idx"]
        selected_img = image_paths[selected_img_idx]
        for idx,img_path in enumerate(image_paths):
            is_selected = (idx==selected_img_idx)
            self.grid.add_widget(ImgSelectScrollItem(self.state_dict, idx, img_path, is_selected))



class ExportOptionBox(ColBoxLayout):
    def __init__(self, state_dict, img_basename, size_hint=(1.0,1.0)):
        super().__init__(cp.EXPORT_OPTION_BG, orientation='horizontal', size_hint=size_hint, padding=[5,0,5,0])
        self.state_dict = state_dict
        self.img_basename = img_basename
        self.add_widget(Label(text="Export", size_hint=(None,None), color=cp.WHITE_TEXT, width=50, height=30))
        # add checkbox
        should_export = self.state_dict["pose_dict"][img_basename]["export"]
        self.export_checkbox = CheckBox(size_hint=(None,None), size=(30,30), active=should_export)
        self.export_checkbox.bind(active=self.on_press)
        self.add_widget(self.export_checkbox)

    def on_press(self, instance, value):
        self.state_dict["pose_dict"][self.img_basename]["export"] = value


class ImgSelectScrollItem(BoxLayout):
    def __init__(self, state_dict, img_idx, img_path, is_selected):
        super().__init__(orientation='horizontal', size_hint=(1.0,None), height=35, padding=[0,0,0,5])
        self.state_dict = state_dict
        self.img_idx = img_idx
        self.img_basename = os.path.basename(img_path)

        if is_selected:
            bg_color = cp.AR_SELECTED_IMG_BG
        else:
            bg_color = cp.PNP_SIDEBAR_CORR

        self.inner_box = ColBoxLayout(bg_color, size_hint=(1.0,1.0))
        self.add_widget(self.inner_box)

        
        select_btn = Button(text="Select", size_hint=(0.3,1.0))
        select_btn.bind(on_press=self.select_img_cb)
        self.inner_box.add_widget(select_btn)
        self.inner_box.add_widget(SidebarHeader(text=self.img_basename, font_size=16))
        self.inner_box.add_widget(ExportOptionBox(self.state_dict, self.img_basename, size_hint=(0.4,1.0)))

    def select_img_cb(self, btn_instance=None):
        self.state_dict["aruco"]["selected_img_idx"] = self.img_idx
        self.state_dict["functions"]["render_aruco_img_select"]()
        self.state_dict["functions"]["update_aruco_detect_display"]()
        self.state_dict["functions"]["render_aruco_graphopt_display_img"]()
        self.state_dict["functions"]["update_gan_remove_aruco_display"]()







class ArucoGraphoptWorkspace(ColBoxLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, orientation="vertical")
        self.state_dict = state_dict
        hori1 = BoxLayout(orientation='horizontal', size_hint=(1.0,0.5))
        self.add_widget(hori1)
        self.add_widget(ColBoxLayout(cp.FS_DELIMITER, size_hint=(1.0,None), height=20))
        hori2 = BoxLayout(orientation='horizontal', size_hint=(1.0,0.5))
        self.add_widget(hori2)
        hori2.add_widget(ArucoOptPoseDisplay(state_dict))

        hori1.add_widget(ArucoDetectDisplay(state_dict))
        hori1.add_widget(GanRemoveArucoDisplay(state_dict))

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
        rgb_img = draw_markers_board(rgb_img, K,"DICT_APRILTAG_16h5")
        return rgb_img

    def update_aruco_detect_display(self):
        rgb_img = self.get_aruco_display_img()
        self.update_texture(rgb_img)



class ArucoOptPoseDisplay(DragZoomImage):
    def __init__(self, state_dict):
        self.state_dict = state_dict
        rgb_shape = self.state_dict["scene"]["orig_img_size"]
        rgb_img = np.zeros((rgb_shape[0],rgb_shape[1],3))
        super().__init__(rgb_img, cp.FILESELECT_BG)
        self.state_dict["functions"]["render_aruco_graphopt_display_img"] = self.render_aruco_graphopt_display_img
        self.state_dict["functions"]["show_overlap_graphopt"] = self.show_overlap_graphopt

    def render_aruco_graphopt_display_img(self):
        graphopt_is_solved = self.state_dict["aruco"]["graphopt_is_solved"]
        if graphopt_is_solved:
            img_dir = self.state_dict["paths"]["image_dir"]
            image_paths = get_image_paths_from_dir(img_dir)
            image_paths.sort()
            selected_img_idx = self.state_dict["aruco"]["selected_img_idx"]
            selected_img = image_paths[selected_img_idx]
            rgb_img = read_rgb(selected_img)
            selected_img_basename = os.path.basename(selected_img)
            obj_path = self.state_dict["paths"]["selected_model"]
            K = self.state_dict["scene"]["cam_K"]
            T_CO = self.state_dict["pose_dict"][selected_img_basename]["T_CO_opt"]
            img_size = self.state_dict["scene"]["orig_img_size"]
            rend_rgb, rend_depth = render_scene(obj_path, T_CO, K, img_size)
            self.state_dict["aruco"]["rend_img_graphopt"] = rend_rgb
            self.state_dict["aruco"]["rend_depth_graphopt"] = rend_depth
            self.state_dict["aruco"]["rgb_img_graphopt"] = rgb_img
            self.show_overlap_graphopt()

    def show_overlap_graphopt(self):
        graphopt_is_solved = self.state_dict["aruco"]["graphopt_is_solved"]
        if graphopt_is_solved:
            rgb_img = self.state_dict["aruco"]["rgb_img_graphopt"]
            rend_rgb = self.state_dict["aruco"]["rend_img_graphopt"]
            depth = self.state_dict["aruco"]["rend_depth_graphopt"]
            mask = np.where(depth>0, 1, 0)
            blend_alpha = self.state_dict["aruco"]["blend_alpha"]
            overlap = blend_imgs(rgb_img, rend_rgb, mask, blend_alpha)
            self.set_texture(overlap)


class GanRemoveArucoDisplay(BoxLayout):
    def __init__(self, state_dict):
        self.state_dict = state_dict
        super().__init__(orientation="vertical")
        self.update_gan_remove_aruco_display()
        self.state_dict["functions"]["update_gan_remove_aruco_display"] = self.update_gan_remove_aruco_display
        

    def update_gan_remove_aruco_display(self):
        self.clear_widgets()
        remove_aruko_btn = Button(text="Remove Aruco", size_hint=(None,None), size=(300,200), center_x=0.5, center_y=0.5)
        self.add_widget(remove_aruko_btn)
        remove_aruko_btn.bind(on_press=self.on_btn_click)
        # get selected img

    def on_btn_click(self, instance):
        img_dir = self.state_dict["paths"]["image_dir"]
        image_paths = get_image_paths_from_dir(img_dir)
        image_paths.sort()
        selected_img_idx = self.state_dict["aruco"]["selected_img_idx"]
        selected_img = image_paths[selected_img_idx]
        print(selected_img)
        # load img
        rgb_img = read_rgb(selected_img)
        self.clear_widgets()
        aruco_info_dict = self.state_dict["aruco"]["aruco_info"]
        img = remove_aruco_boards_handler(rgb_img, aruco_info_dict)
        zoom_img_handler = DragZoomImage(img, cp.FILESELECT_BG)
        self.add_widget(zoom_img_handler)





class ArucoGraphoptGUI(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='horizontal')
        self.state_dict = state_dict
        self.add_widget(ArucoGraphoptSidebar(state_dict))
        self.add_widget(ArucoGraphoptWorkspace(state_dict))
        self.state_dict["functions"]["render_aruco_graphopt_display_img"]()


