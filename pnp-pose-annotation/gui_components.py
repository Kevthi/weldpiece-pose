import numpy as np
import cv2

from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.uix.label import Label

class ColorProfile():
    def __init__(self):
        """
        Initializes the ColorProfile class for the GUI.
        Intensity of the color in RGB = [R,G,B]/256.
        """
        KIVY_DEF_BTN_COL = np.array([88,88,88])/256

        NTNU_BLUE = [0,80/255.0,158/255.0]
        NTNU_BLUE_LIGHT = [179/255.0,217/255.0, 255/255.0]
        BLACK = [0.0,0.0,0.0]
        WHITE = [1.0,1.0,1.0]
        LIGHT_GRAY = [0.95,0.95,0.95]
        GRAY = [0.85,0.85,0.85]
        DARKER_GRAY = [0.75,0.75,0.75]
        DARKEST_GRAY = [0.60,0.60,0.60]
        BLACK_GRAY = [0.3,0.3,0.3]

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
        self.FS_DELIMITER = self.SIDEBAR
        self.FS_CARD_BUTTON = np.array(NTNU_BLUE)/KIVY_DEF_BTN_COL
        self.FS_SELECTED_CARD = np.array(NTNU_BLUE_LIGHT)
        #PNP
        self.PNP_SLIDER_BG = NTNU_BLUE
        self.PNP_SIDEBAR_CORR = DARKER_GRAY
        self.PNP_CORR_WIN_BG = GRAY
        self.PNP_SAVE_POSE_BTN =np.array(NTNU_BLUE)/KIVY_DEF_BTN_COL 

        #ARUCO
        self.AR_SELECTED_IMG_BG = NTNU_BLUE_LIGHT
        self.EXPORT_OPTION_BG = BLACK_GRAY
        


        
#Here we create an instance of the ColorProfile class, and assign
color_profile = ColorProfile()
cp = color_profile



class ColBoxLayout(BoxLayout):
    """
    ColBoxLayout is a custom BoxLayout that supports setting a background color.

    Example of usage:
    col_box_layout = ColBoxLayout(bg_color=[0.1, 0.2, 0.3], orientation='vertical', size_hint=(1.0, 0.5))
    The **kwargs are passed to the BoxLayout constructor with the necessary arguments.

    Attributes:
        bg_color (ColorProfile): The background color of the layout.
        rect (Rectangle): The rectangle representing the background color.
    """

    def __init__(self, bg_color: ColorProfile, **kwargs):
        """
        Initializes the ColBoxLayout with a background color and binds size and position updates.

        Args:
            bg_color (ColorProfile): The background color of the layout.
            **kwargs: Additional keyword arguments passed to the BoxLayout constructor.
        """
        super().__init__(**kwargs)
        self.bind(size=self._update_rect, pos=self._update_rect)

        # Setting the background color
        with self.canvas.before:
            Color(bg_color[0], bg_color[1], bg_color[2], 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

    def _update_rect(self, instance, value):
        """
        Updates the position and size of the background rectangle.

        Args:
            instance: The instance of the layout.
            value: The new value of the property that triggered the update.
        """
        self.rect.pos = instance.pos
        self.rect.size = instance.size

    def update_color(self, bg_color):
        """
        Updates the background color of the layout.

        Args:
            bg_color (ColorProfile): The new background color.
        """
        with self.canvas.before:
            Color(bg_color[0], bg_color[1], bg_color[2], 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

class TextureImage(Image):
    """
    TextureImage is a custom Image widget that supports displaying an RGB image.

    Attributes:
        texture (Texture): The texture created from the RGB image.
    """

    def __init__(self, rgb_img : cv2):
        """
        Initializes the TextureImage with an RGB image.

        Args:
            rgb_img: The RGB image to be displayed.
        """
        super().__init__(keep_ratio=True, allow_stretch=True)
        rgb_img_flip = cv2.flip(rgb_img, 0)
        buf = rgb_img_flip.tobytes()
        self.texture = Texture.create(size=(rgb_img.shape[1], rgb_img.shape[0]), colorfmt='rgb')
        self.texture.blit_buffer(buf, bufferfmt='ubyte', colorfmt='rgb')

    def update_texture(self, rgb_img):
        """
        Updates the texture with a new RGB image.

        Args:
            rgb_img: The new RGB image to be displayed.
        """
        rgb_img_flip = cv2.flip(rgb_img, 0)
        buf = rgb_img_flip.tobytes()
        self.texture.blit_buffer(buf, bufferfmt='ubyte', colorfmt='rgb')




        


class ColGridLayout(GridLayout):
    """
    ColGridLayout is a custom GridLayout that supports setting a background color.

    Args:
        bg_color (list): The background color of the layout.
        **kwargs: Additional keyword arguments passed to the GridLayout constructor.
    """
    def __init__(self, bg_color, **kwargs):
        super().__init__(**kwargs)
        self.bind(size=self._update_rect, pos=self._update_rect)

        # Setting the background color
        with self.canvas.before:
            Color(bg_color[0], bg_color[1], bg_color[2], 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

    def _update_rect(self, instance, value):
        """
        Updates the position and size of the background rectangle.

        Args:
            instance: The instance of the layout.
            value: The new value of the property that triggered the update.
        """
        self.rect.pos = instance.pos
        self.rect.size = instance.size


class ColAnchorLayout(AnchorLayout):
    """
    ColAnchorLayout is a custom AnchorLayout that supports setting a background color.

    Args:
        bg_color (list): The background color of the layout.
        **kwargs: Additional keyword arguments passed to the AnchorLayout constructor.
    """
    def __init__(self, bg_color, **kwargs):
        super().__init__(**kwargs)
        self.bind(size=self._update_rect, pos=self._update_rect)

        # Setting the background color
        with self.canvas.before:
            Color(bg_color[0], bg_color[1], bg_color[2], 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)

    def _update_rect(self, instance, value):
        """
        Updates the position and size of the background rectangle.

        Args:
            instance: The instance of the layout.
            value: The new value of the property that triggered the update.
        """
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

class SidebarHeader(AnchorLayout):
    def __init__(self, text="", height=30, font_size=20, color=(0.0,0.0,0.0)):
        super().__init__(size_hint=(1.0, None), height=height,anchor_x='center', anchor_y='center')
        label = Label(text=text, color=color, font_size=font_size)
        self.add_widget(label)









