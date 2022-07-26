from gui_components import Sidebar, SidebarButton
from gui_utils import ask_directory, ask_file

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout

from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color

from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.scatter import Scatter, ScatterPlane

from kivy.graphics.transformation import Matrix


import cv2
import numpy as np

def generate_texture():
    """Generate random numpy array `500x500` as iamge, use cv2 to change image, and convert to Texture."""

    # numpy array
    img = np.random.randint(0, 256, size=(500, 500, 3), dtype=np.uint8)
    cv2.circle(img, (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
    data = img.tobytes()

    # texture
    texture = Texture.create(size=(500, 500), colorfmt="rgb")
    texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
    texture.mag_filter = 'nearest'
    texture.min_filter = 'nearest'

    return texture



class MyScatterPlane(ScatterPlane):
    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    mat = Matrix().scale(.9,.9,.9)
                    self.apply_transform(mat, anchor=touch.pos)
                elif touch.button == 'scrollup':
                    mat = Matrix().scale(1.1,1.1,1.1)
                    self.apply_transform(mat, anchor=touch.pos)
        return super().on_touch_up(touch)

class DragZoomImageHandler(Image):
    def __init__(self, rgb_img):
        super().__init__(allow_stretch=True)
        print("dfsfd")
        self.set_texture(rgb_img)
        self.orig_width = 500
        self.current_zoom = 0


    """
    def on_touch_down(self, touch):
        print("Touch", touch)
        print(touch == 'scrolldown')
    """
    def collide_point(self, pos_x, pos_y):
        wpos_x, wpos_y = self.pos
        width,height = self.norm_image_size
        within_x = (pos_x > wpos_x and pos_x < (wpos_x+width))
        within_y = (pos_y > wpos_y and pos_y < (wpos_y+height))
        return (within_x and within_y)

    def get_texture_bounding_box(self):
        outer_wx = self.width
        outer_wy = self.height
        outer_posx, outer_posy = self.pos

        center_x = outer_posx + outer_wx/2
        center_y = outer_posy + outer_wy/2

        inner_wx, inner_wy = self.norm_image_size
        
        print("center x", center_x, "center_y", center_y)



    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            print("Touch pos", touch.pos)
            print("Collide point", self.collide_point(touch.pos[0], touch.pos[1]))
            print("width", self.width)
            print("width", self.height)
            print("widget pos", self.pos)
            print("norm image size", self.norm_image_size)
            self.get_texture_bounding_box()
            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    mat = Matrix().scale(.9,.9,.9)
                    min_crop = self.current_zoom
                    max_crop = self.orig_width-min_crop*2
                    if (self.current_zoom + 20)>500:
                        self.current_zoom = 490
                    else:
                        self.current_zoom += 20
                    self.texture = self.orig_texture.get_region(min_crop, min_crop, max_crop, max_crop)
                    #self.apply_transform(mat, anchor=touch.pos)
                elif touch.button == 'scrollup':
                    min_crop = self.current_zoom
                    max_crop = self.orig_width-min_crop*2
                    if (self.current_zoom - 20)<0:
                        self.current_zoom = 0
                    else:
                        self.current_zoom -= 20
                    self.texture = self.orig_texture.get_region(min_crop, min_crop, max_crop, max_crop)
        return super().on_touch_up(touch)

    def set_texture(self, rgb_img):
        print("shape",rgb_img.shape)
        data = rgb_img.tobytes()
        # texture
        texture = Texture.create(size=(rgb_img.shape[0], rgb_img.shape[1]), colorfmt="rgb")
        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
        texture.mag_filter = 'nearest'
        texture.min_filter = 'nearest'
        self.texture = texture
        self.orig_texture = texture
        self.orig_size = self.texture.size
        



class DragZoomImage(AnchorLayout):
    def __init__(self, texture, **kwargs):
        super().__init__(anchor_x='center',  anchor_y='center', **kwargs)
        image = DragZoomImageHandler(texture)
        self.add_widget(image)





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
        self.add_widget(DragZoomImage(img, padding=0))
        self.add_widget(DragZoomImage(img, padding=0))






