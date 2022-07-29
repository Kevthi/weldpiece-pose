import os
from renderer import render_thumbnail

from gui_components import Sidebar, SidebarButton, ColBoxLayout, ColGridLayout, ColAnchorLayout, ColScrollView
from gui_components import TextureImage
from gui_utils import ask_directory, ask_file, get_image_paths_from_dir, read_rgb, get_files_from_dir
from gui_components import color_profile as cp

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.anchorlayout import AnchorLayout
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.uix.label import Label



class FileSelectCard(ColBoxLayout):
    def __init__(self, rgb_img, path):
        super().__init__(cp.CARD_BG, orientation='vertical', height=400, size_hint=(0.3,None), size_hint_max_x=400, padding=10)
        #img = Image(source=img_path, size_hint=(1.0,1.0), allow_stretch=True, keep_ratio=True)
        self.path = path
        img = TextureImage(rgb_img)
        self.add_widget(img)
        img_name = os.path.basename(path)
        print("img_name", img_name)
        self.add_widget(Label(text=img_name, size_hint=(1.0, None), height=30, color=cp.BLACK_TEXT))
        btn = Button(text="Select", size_hint=(1.0,None), height=30, background_color=cp.FS_CARD_BUTTON)
        self.add_widget(btn)
        btn.bind(on_press=self.on_select)

    def on_select(self, instance):
        print("Selected")
        self.parent.reset_card_colors()
        self.update_color(cp.FS_SELECTED_CARD)

class CardGrid(ColGridLayout):
    def __init__(self):
        super().__init__(cp.FILESELECT_BG, cols=3, size_hint_y=None, padding=[40,40,40,0], spacing=40)
        self.bind(minimum_height=self.setter('height'))
        self.cards = []

    def add_card_widget(self, card):
        self.cards.append(card)
        self.add_widget(card)

    def reset_card_colors(self):
        for card in self.cards:
            card.update_color(cp.CARD_BG)

        


class FileSelectSidebar(Sidebar):
    def __init__(self):
        super().__init__()
        self.add_widget(SidebarButton(height=50))
        self.add_widget(Widget())




class ImgSelectScroll(ColScrollView):
    def __init__(self, img_paths):
        super().__init__(cp.FILESELECT_BG,do_scroll_y=True, bar_width=20, bar_color=cp.SCROLLBAR, bar_inactive_color=cp.SCROLLBAR)
        self.card_grid = CardGrid()
        for img_path in img_paths:
            rgb_img = read_rgb(img_path)
            fs_card = FileSelectCard(rgb_img, img_path)
            self.card_grid.add_card_widget(fs_card)
        self.add_widget(self.card_grid)

class SelectDirectory(ColAnchorLayout):
    def __init__(self, text, callback):
        super().__init__(cp.FILESELECT_BG, anchor_x='center', anchor_y='center')
        dir_select_btn = Button(text=text, font_size=24, size_hint=(None,None), width=450, height=150, background_color=cp.DIR_SELECT_BTN, color=cp.WHITE_TEXT)
        dir_select_btn.bind(on_press=self.select_dir_cb)
        self.add_widget(dir_select_btn)
        self.callback = callback


    def select_dir_cb(self, instance):
        img_dir = ask_directory()
        self.callback(img_dir)


class CameraInfoSelectBox(AnchorLayout):
    def __init__(self, state_dict, camera_info_path):
        super().__init__(anchor_x='center', anchor_y='center', size_hint=(1.0,None), height=50)
        self.camera_info_path = camera_info_path
        self.state_dict = state_dict
        self.btn = Button(text=camera_info_path, size_hint=(0.5,1.0))
        self.add_widget(self.btn)

    def on_select_path(self, instance):
        self.parent.parent.remove_camera_info_select()



class CameraInfoSelectFile(AnchorLayout):
    def __init__(self, state_dict):
        super().__init__(anchor_x='center', anchor_y='center', size_hint=(1.0,1.0))
        self.state_dict = state_dict
        self.select_file_btn = Button(text="Select camera info file (JSON/NPY)", background_color=cp.DIR_SELECT_BTN, width=450, height=150, size_hint=(None,None), font_size=20)
        self.select_file_btn.bind(on_press=self.on_select_path)
        self.add_widget(self.select_file_btn)


    def on_select_path(self, instance):
        camera_info_path = ask_file()
        self.state_dict["paths"]["camera_info_path"] = camera_info_path
        self.state_dict["functions"]["on_files_selected"]()
        self.parent.parent.remove_camera_info_select()





class CameraInfoSelectBoxMain(ColBoxLayout):
    def __init__(self, state_dict):
        super().__init__(cp.FILESELECT_BG, orientation='vertical', size_hint=(1.0,0.3))

        self.add_widget(CameraInfoSelectFile(state_dict))
        self.add_widget(ColBoxLayout(cp.FS_DELIMITER, size_hint=(1.0,None), height=20))


        


class ImgSelectMainWin(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='vertical')
        self.state_dict = state_dict
        if state_dict["paths"]["image_dir"] is None:
            self.add_widget(SelectDirectory("Select image directory", self.image_dir_select_cb))
        else:
            img_paths = get_image_paths_from_dir(state_dict["paths"]["image_dir"])
            scroll_view = ImgSelectScroll(img_paths)
            if state_dict["paths"]["camera_info_path"] is None:
                self.camera_info_select = CameraInfoSelectBoxMain(state_dict)
                self.add_widget(self.camera_info_select)

            self.add_widget(scroll_view)


    def remove_camera_info_select(self):
        self.remove_widget(self.camera_info_select)


    def image_dir_select_cb(self, img_dir):
        if type(img_dir) is not str:
            return
        if not os.path.isdir(img_dir):
            return
        self.state_dict["paths"]["image_dir"] = img_dir
        self.state_dict["functions"]["set_fs_tab"]()

class ModelSelectMainWin(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='vertical')
        self.state_dict = state_dict
        if state_dict["paths"]["3dmodel_dir"] is None:
            self.select_model_dir = SelectDirectory("Select 3d model directory", self.model_dir_select_cb)
            self.add_widget(self.select_model_dir)
        else:
            model_dir = state_dict["paths"]["3dmodel_dir"]
            self.add_scroll_view(model_dir)

    def model_dir_select_cb(self, model_dir):
        if type(model_dir) is not str:
            return
        if not os.path.isdir(model_dir):
            return
        self.state_dict["paths"]["3dmodel_dir"] = model_dir
        self.remove_widget(self.select_model_dir)
        self.add_scroll_view(model_dir)

    def add_scroll_view(self, model_dir):
        model_paths = get_files_from_dir(model_dir, "ply")
        scroll_view = ModelSelectScroll(model_paths)
        self.add_widget(scroll_view)



class ModelSelectScroll(ColScrollView):
    def __init__(self, model_paths):
        super().__init__(cp.FILESELECT_BG,do_scroll_y=True, bar_width=20, bar_color=cp.SCROLLBAR, bar_inactive_color=cp.SCROLLBAR)
        self.card_grid = CardGrid()
        for model_path in model_paths:
            rgb_img = render_thumbnail(model_path)
            fs_card = FileSelectCard(rgb_img, model_path)
            self.card_grid.add_card_widget(fs_card)
        self.add_widget(self.card_grid)






class FileSelectGUI(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='horizontal')
        self.state_dict = state_dict
        self.state_dict["functions"]["on_files_selected"] = self.on_files_selected
        self.add_widget(FileSelectSidebar())
        self.vert_layout = BoxLayout(orientation='vertical')
        self.horiz_layout = BoxLayout(orientation='horizontal', size_hint=(1.0,0.8))
        self.vert_layout.add_widget(self.horiz_layout)
        self.add_widget(self.vert_layout)

        #self.vert_layout.add_widget(BoxLayout(size_hint=(1.0,0.2)))
        self.horiz_layout.add_widget(ImgSelectMainWin(state_dict))
        self.horiz_layout.add_widget(ColBoxLayout(cp.FS_DELIMITER, size_hint=(None,1.0), width=20))
        self.horiz_layout.add_widget(ModelSelectMainWin(state_dict))

    def on_files_selected(self):
        print("on files selected")
        img_selected = (self.state_dict["paths"]["selected_img"] is not None)
        model_selected = self.state_dict["paths"]["selected_model"] is not None
        camera_info_selected = self.state_dict["paths"]["camera_info_path"] is not None
        if (img_selected and model_selected and camera_info_selected):
            print("All selected")
















