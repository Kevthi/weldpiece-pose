
from gui_init_pose import InitPoseGUI
from gui_file_select import FileSelectGUI
from gui_pnp import PnPGUI
from gui_components import ColBoxLayout
from gui_components import color_profile as cp

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color

SB_COL = [.1,.1,.1]
SB_COL_BTN = [.3,.3,.3]
DARKEST = [.05,.05,.05]
DARKEST_BTN = [.15,.15,.15]
GRAY_TEXT_COLOR = [.85,.85,.85]



class Topbar(ColBoxLayout):
    def __init__(self, tab_dict, active_tab=None):
        super().__init__(cp.TOPBAR, orientation='horizontal', size_hint=(1.0, None), height=45) 
        for tab_name in tab_dict:
            if tab_name == active_tab:
                bg_color = cp.TOPBAR_BTN_ACTIVE
                text_color=cp.BLACK_TEXT
            else:
                bg_color = cp.TOPBAR_BTN_INACTIVE
                text_color=cp.WHITE_TEXT
            cb_func = tab_dict[tab_name]["callback"]
            tab_btn = Button(text=tab_name, background_normal='',size_hint=(None, 1.0), width=300, background_color=bg_color, font_size=20, color=text_color)
            tab_btn.bind(on_press=cb_func)
            self.add_widget(tab_btn)
            self.add_widget(ColBoxLayout(cp.TOPBAR_BREAK,size_hint=(None,1.0), width=2))



class BelowTopbar(ColBoxLayout):
    def __init__(self):
        super().__init__(cp.SIDEBAR, orientation='horizontal', size_hint=(1.0, None), height=60) 


def init_state_dict():
    state_dict = {
        "functions":{
            "set_fs_tab":None,
            "set_ip_tab":None,
            "set_pnp_tab":None,
            "rerender":None,
            "on_add_corr":None,
            "draw_corrs":None,
        },
        "paths":{
            "image_dir":None,
            "3dmodel_dir":None,
            "selected_img":None,
            "selected_model":None,
            "camera_info_path":None,
        },
        "scene":{
            "cam_K":None,
            "T_wc":None,
            "orig_img_size":None,
        },
        "pnp":{
            "corresps":[],
            "img_select":None,
            "rend_select":None,
            "normal_render":False,
            "cam_img":None,
            "rend_img":None,
            "rend_depth":None,
            "marker_size":5,
        },
    }
    return state_dict




class GUIMain(App):
    def build(self):
        self.FILE_TABNAME = "Select Files"
        self.INIT_POSE_TABNAME = "Initialize pose"
        self.PNP_TABNAME = "Solve PnP"

        self.state_dict = init_state_dict()
        self.state_dict["functions"]["set_fs_tab"] = self.set_file_select_active
        self.state_dict["functions"]["set_ip_tab"] = self.set_pose_init_active
        self.state_dict["functions"]["set_pnp_tab"] = self.set_pnp_active

        # temp for testing
        self.state_dict["paths"]["selected_img"] = "/home/ola/projects/weldpiece-pose-datasets/ds-projects/office-corner/captures/corner1-undist.png"
        self.state_dict["paths"]["selected_model"] = "/home/ola/projects/weldpiece-pose-datasets/pnp-pose-annotation/corner.ply"
        self.state_dict["paths"]["camera_info_path"] = "/home/ola/projects/weldpiece-pose-datasets/ds-projects/office-corner/captures/info.json"




        self.tab_dict = {
            self.FILE_TABNAME: {
                "callback": self.set_file_select_active,
                "is_accessible":True,
            },
            self.INIT_POSE_TABNAME: {
                "callback": self.set_pose_init_active,
                "is_accessible":True,
            },
            self.PNP_TABNAME: {
                "callback":self.set_pnp_active,
                "is_accessible":True,
            }
        }


        self.gui_window = BoxLayout(orientation='vertical')
        self.set_file_select_active()
        return self.gui_window

    def rerender_mainwin(self, tab_dict, active_tab=None):
        self.gui_window.clear_widgets()
        topbar = Topbar(tab_dict, active_tab)
        self.gui_window.add_widget(topbar)
        below_topbar = BelowTopbar()
        self.gui_window.add_widget(below_topbar)
        main_win = BoxLayout(orientation='horizontal')
        self.gui_window.add_widget(main_win)
        return main_win

    def set_file_select_active(self, btn_obj=None):
        file_select = FileSelectGUI(self.state_dict)
        main_win = self.rerender_mainwin(self.tab_dict, self.FILE_TABNAME)
        main_win.add_widget(file_select)


    def set_pose_init_active(self, btn_obj=None):
        pose_init = InitPoseGUI(self.state_dict)
        main_win = self.rerender_mainwin(self.tab_dict, self.INIT_POSE_TABNAME)
        main_win.add_widget(pose_init)

    def set_pnp_active(self, btn_obj=None):
        pnp_gui = PnPGUI(self.state_dict)
        main_win = self.rerender_mainwin(self.tab_dict, self.PNP_TABNAME)
        main_win.add_widget(pnp_gui)


if __name__ == '__main__':
    from kivy.core.window import Window
    Window.maximize()
    GUIMain().run()


