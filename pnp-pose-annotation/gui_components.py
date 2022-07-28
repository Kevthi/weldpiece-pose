
import numpy as np
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color


from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView

class ColorProfile():
    def __init__(self):

        KIVY_DEF_BTN_COL = np.array([88,88,88])/256

        NTNU_BLUE = [0,80/255.0,158/255.0]
        BLACK = [0.0,0.0,0.0]
        WHITE = [1.0,1.0,1.0]
        LIGHT_GRAY = [0.95,0.95,0.95]
        GRAY = [0.85,0.85,0.85]
        DARKER_GRAY = [0.75,0.75,0.75]
        DARKEST_GRAY = [0.60,0.60,0.60]

        self.BLACK_TEXT = BLACK
        self.WHITE_TEXT = WHITE

        self.SCROLLBAR = DARKEST_GRAY
        # SIDEBAR
        self.SIDEBAR = LIGHT_GRAY
        self.SIDEBAR_BTN=[1.95,1.95,1.95]
        self.SIDEBAR_BTN_TEXT = BLACK
        # TOPBAR
        self.TOPBAR = NTNU_BLUE
        self.TOPBAR_BREAK = np.array(NTNU_BLUE)+0.1
        self.TOPBAR_BTN_INACTIVE = np.array(NTNU_BLUE)
        self.TOPBAR_BTN_ACTIVE = self.SIDEBAR
        self.TOPBAR_BTN_TEXT = BLACK
        #FILESELECT
        #self.FILESELECT_BG_SCROLLBAR = WHITE
        self.FILESELECT_BG = GRAY
        self.CARD_BG = LIGHT_GRAY
        self.DIR_SELECT_BTN = np.array(NTNU_BLUE)/KIVY_DEF_BTN_COL
        
        

color_profile = ColorProfile()
cp = color_profile

class ColBoxLayout(BoxLayout):
    def __init__(self, bg_color, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            size=self._update_rect,
            pos=self._update_rect
        )


        with self.canvas.before:
            Color(bg_color[0], bg_color[1], bg_color[2], 1)
            self.rect = Rectangle(
                size=self.size,
                pos=self.pos
            )

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

class ColGridLayout(GridLayout):
    def __init__(self, bg_color, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            size=self._update_rect,
            pos=self._update_rect
        )


        with self.canvas.before:
            Color(bg_color[0], bg_color[1], bg_color[2], 1)
            self.rect = Rectangle(
                size=self.size,
                pos=self.pos
            )

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size


class ColAnchorLayout(AnchorLayout):
    def __init__(self, bg_color, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            size=self._update_rect,
            pos=self._update_rect
        )


        with self.canvas.before:
            Color(bg_color[0], bg_color[1], bg_color[2], 1)
            self.rect = Rectangle(
                size=self.size,
                pos=self.pos
            )

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size

class ColScrollView(ScrollView):
    def __init__(self, bg_color, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            size=self._update_rect,
            pos=self._update_rect
        )


        with self.canvas.before:
            Color(bg_color[0], bg_color[1], bg_color[2], 1)
            self.rect = Rectangle(
                size=self.size,
                pos=self.pos
            )

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size


class Sidebar(ColBoxLayout):
    def __init__(self):
        super().__init__(color_profile.SIDEBAR, orientation='vertical', size_hint=(None, 1.0), width=400, padding=[0, 100, 0, 0])




class SidebarButton(Button):
    def __init__(self, text="", height=30, callback=None):
        super().__init__(size_hint=(1.0, None), text=text, height=height, background_color=cp.SIDEBAR_BTN, color=cp.SIDEBAR_BTN_TEXT)
        if callback is not None:
            self.bind(callback)







