import os

from gui_components import Sidebar, SidebarButton, ColBoxLayout, ColGridLayout, ColAnchorLayout, ColScrollView
from gui_utils import ask_directory, ask_file
from gui_components import color_profile as cp

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.uix.label import Label

class FileSelectCard(ColBoxLayout):
    def __init__(self, img_path):
        super().__init__(cp.CARD_BG, orientation='vertical', height=400, size_hint=(0.3,None), size_hint_max_x=400, padding=10)
        img = Image(source=img_path, size_hint=(1.0,1.0), allow_stretch=True, keep_ratio=True)
        self.add_widget(img)
        img_name = os.path.basename(img_path)
        print("img_name", img_name)
        self.add_widget(Label(text=img_name, size_hint=(1.0, None), height=30, color=cp.BLACK_TEXT))
        btn = Button(text="Select", size_hint=(1.0,None), height=30, background_color=cp.SIDEBAR_BTN)
        self.add_widget(btn)

class CardGrid(ColGridLayout):
    def __init__(self):
        super().__init__(cp.FILESELECT_BG, cols=3, size_hint_y=None, padding=[40,40,40,0], spacing=40)
        self.bind(minimum_height=self.setter('height'))
        


class FileSelectSidebar(Sidebar):
    def __init__(self):
        super().__init__()
        self.add_widget(SidebarButton(height=50))
        self.add_widget(Widget())

class FileSelectScroll(ColScrollView):
    def __init__(self, img_paths):
        super().__init__(cp.FILESELECT_BG,do_scroll_y=True, bar_width=20, bar_color=cp.SCROLLBAR, bar_inactive_color=cp.SCROLLBAR)
        self.card_grid = CardGrid()
        for img_path in img_paths:
            fs_card = FileSelectCard(img_path)
            self.card_grid.add_widget(fs_card)
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




class FileSelectGUI(BoxLayout):
    def __init__(self, state_dict):
        super().__init__(orientation='horizontal')
        self.state_dict = state_dict
        self.add_widget(FileSelectSidebar())

        if state_dict["image_dir"] is None:
            self.add_widget(SelectDirectory("Select image directory", self.image_dir_select_cb))
        else:
            img_paths = self.get_image_paths_from_dir(state_dict["image_dir"])
            scroll_view = FileSelectScroll(img_paths)
            self.add_widget(scroll_view)

        if state_dict["3dmodel_dir"] is None:
            self.add_widget(SelectDirectory("Select 3d model directory", self.model_dir_select_cb))
        else:
            scroll_view = FileSelectScroll()
            self.add_widget(scroll_view)

    def image_dir_select_cb(self, img_dir):
        if type(img_dir) is not str:
            return
        if not os.path.isdir(img_dir):
            return
        print("img dir", img_dir)
        self.state_dict["image_dir"] = img_dir
        print(self.get_image_paths_from_dir(img_dir))
        self.state_dict["functions"]["set_fs_tab"]()

    def model_dir_select_cb(self, model_dir):
        if type(model_dir) is not str:
            return
        if not os.path.isdir(model_dir):
            return
        print("model dir", model_dir)

    @staticmethod
    def get_image_paths_from_dir(img_dir, allowed_formats=("png", "jpg")):
        return [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) if filename.endswith((allowed_formats))]











