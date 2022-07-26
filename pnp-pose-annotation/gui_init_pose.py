from gui_components import Sidebar, SidebarButton
from gui_utils import ask_directory, ask_file

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color


class InitPoseSidebar(Sidebar):
    def __init__(self):
        super().__init__()
        self.add_widget(SidebarButton(text="AAA", height=50))
        self.add_widget(SidebarButton(height=50))
        self.add_widget(Widget())


class InitPoseGUI(BoxLayout):
    def __init__(self):
        super().__init__(orientation='horizontal')
        self.add_widget(InitPoseSidebar())








