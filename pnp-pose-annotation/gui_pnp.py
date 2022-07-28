from gui_components import Sidebar, SidebarButton
from gui_utils import ask_directory, ask_file

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
    def __init__(self):
        super().__init__()
        self.add_widget(SidebarButton(text="Add corr", height=50))
        self.add_widget(SidebarButton(height=50))
        self.add_widget(Widget())



class PnPGUI(BoxLayout):
    def __init__(self):
        super().__init__(orientation='horizontal')
        self.add_widget(PnPSidebar())
        #float_layout = FloatLayout()
        #anchor = AnchorLayout(anchor_x="center", anchor_y="center")
        #img = MyImage()
        #anchor.add_widget(img)
        img = np.random.randint(0, 30, size=(1001, 1001, 3), dtype=np.uint8)
        cv2.circle(img, (img.shape[1]//2, img.shape[0]//2), 3, 255, -1)
        self.image1 = DragZoomImage(img, [1.0,1.0,1.0], padding=0)
        self.image1.bind_image_cb(self.img_callback)
        self.add_widget(self.image1)

        self.add_widget(DragZoomImage(img, padding=10, background_color=[1.0,1.0,1.0]))

    def img_callback(self, pos):
        print("print pos from cb", pos)
        #img = np.random.randint(0, 30, size=(1001, 1001, 3), dtype=np.uint8)
        img = cv2.cvtColor(cv2.imread("corner1-undist.png"), cv2.COLOR_BGR2RGB)
        img[pos[0],pos[1],:] = np.array([255,0,0])
        self.image1.set_texture(img)








