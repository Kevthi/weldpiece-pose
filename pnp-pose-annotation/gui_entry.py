
from gui_init_pose import InitPoseGUI
from gui_file_select import FileSelectGUI
from gui_pnp import PnPGUI
from gui_components import ColBoxLayout
from gui_components import color_profile as cp
from gui_aruco_graphopt import ArucoGraphoptGUI

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
    """
    Inherent the ColBoxLayout class.
    Topbar class represents the top navigation bar in the GUI.
    Attributes:
        tab_dict (dict): Sets up the tabs and their function binded the them.
        example pnp-tab: {"Solve PnP": self.set_pnp_active}
        active_tab (str): The name of the currently active tab.
    """
    def __init__(self, tab_dict : dict, active_tab : str = None ):
        super().__init__(cp.TOPBAR, orientation='horizontal', size_hint=(1.0, None), height=45) 
        """
        Initializes the Topbar of the gui.
        cp.TOPBAR: Is a own defined class in "gui_components.py" containing the color profile for different parts of the GUI.
        orientation: is a parameter of the BoxLayout class telling how the window should be placed.
        size_hint: is a parameter of the BoxLayout class telling how the window should be placed.
        height: is a parameter of the BoxLayout class telling how the window should be placed.
        """
        #loop check if the tab is active or not 
        for tab_name in tab_dict:
            if(tab_name == active_tab):
                background_color = cp.TOPBAR_BTN_ACTIVE
                text_color = cp.BLACK_TEXT
            else:
                bg_color = cp.TOPBAR_BTN_INACTIVE
                text_color=cp.WHITE_TEXT
            cb_func = tab_dict[tab_name]["callback"]
            #Initiates the button with the name of the tab and the color profile.
            tab_btn = Button(text=tab_name, background_normal='',size_hint=(None, 1.0), width=300, background_color=bg_color, font_size=20, color=text_color)
            tab_btn.bind(on_press=cb_func)
            self.add_widget(tab_btn)
            self.add_widget(ColBoxLayout(cp.TOPBAR_BREAK,size_hint=(None,1.0), width=2))



class BelowTopbar(ColBoxLayout):
    #Creates the below topbar class
    def __init__(self):
        super().__init__(cp.SIDEBAR, orientation='horizontal', size_hint=(1.0, None), height=60) 



def init_state_dict():
    state_dict = {
        "functions":{
            "set_fs_tab":None,
            "set_ip_tab":None,
            "set_pnp_tab":None,
            "set_aruco_graphopt_tab":None,
            "rerender":None,
            "on_add_corr":None,
            "draw_corrs":None,
            "update_sidebar_corrs":None,
            "update_pnp":None,
            "show_overlap_pnp":None,
            "update_aruco_detect_display":None,
            "render_aruco_img_select":None,
            "render_save_pose_btn":None,
            "render_aruco_graphopt_display_img":None,
            "show_overlap_graphopt":None,
            "update_gan_remove_aruco_display":None,
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
            "rend_K":None,
            "marker_size":3,
            "T_WC_pnp":None,
            "rend_img_pnp": None,
            "rend_depth_pnp": None,
            "blend_alpha":0.5,
            "reproj_error":10,
        },
        "aruco":{
            "aruco_info":{
                "aruco_dict_str": "DICT_APRILTAG_16H5",
                "use_board":True,
                "markers_per_board":4,
                "board_shape":(3,3),
                "square_sz":56*1e-3,
                "aruco_sz":35*1e-3,
            },
            "selected_img_idx": 0,
            "graphopt_is_solved":False,
            "blend_alpha":0.5,
            "rend_img_graphopt":None,
            "rend_depth_graphopt":None,
            "rgb_img_graphopt":None,
        },
        "pose_dict":{
        }
    }
    return state_dict




class GUIMain(App):
    def build(self):
        self.FILE_TABNAME = "Select Files"
        self.INIT_POSE_TABNAME = "Initialize pose"
        self.PNP_TABNAME = "Solve PnP"
        self.ARUCO_GRAPHOPT_TABNAME = "Aruco graph optim"


        self.state_dict = init_state_dict()
        self.state_dict["functions"]["set_fs_tab"] = self.set_file_select_active
        self.state_dict["functions"]["set_ip_tab"] = self.set_pose_init_active
        self.state_dict["functions"]["set_pnp_tab"] = self.set_pnp_active
        self.state_dict["functions"]["set_aruco_graphopt_tab"] = self.set_aruco_graphopt_active

        # template for testing
        ds_project = "office-corner-brio-4k-charuco"
        self.state_dict["paths"]["selected_img"] = f'/home/ola/projects/weldpiece-pose-datasets/ds-projects/{ds_project}/captures/img_0-undist.png'
        self.state_dict["paths"]["image_dir"] = f'/home/ola/projects/weldpiece-pose-datasets/ds-projects/{ds_project}/captures'
        self.state_dict["paths"]["selected_model"] = f'/home/ola/projects/weldpiece-pose-datasets/3d-models/corner.ply'
        self.state_dict["paths"]["camera_info_path"] = f'/home/ola/projects/weldpiece-pose-datasets/ds-projects/{ds_project}/captures/info.json'
        """
        import pickle
        with open('pose_dict.pkl', 'rb') as handle:
            pose_dict = pickle.load(handle)

        self.state_dict["pose_dict"] = pose_dict
        """


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
            },
            self.ARUCO_GRAPHOPT_TABNAME:{
                "callback":self.set_aruco_graphopt_active,
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

    def set_aruco_graphopt_active(self, btn_obj=None):
        aruco_graphopt_win = ArucoGraphoptGUI(self.state_dict)
        main_win = self.rerender_mainwin(self.tab_dict, self.ARUCO_GRAPHOPT_TABNAME)
        main_win.add_widget(aruco_graphopt_win)



if __name__ == '__main__':
    from kivy.core.window import Window
    Window.maximize()
    GUIMain().run()


