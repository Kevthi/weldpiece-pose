
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.core.window import Window

from kivy.uix.button import Button
from kivy.uix.widget import Widget

from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Rectangle

import asyncio
from threading import Thread
import time


import cv2
import numpy as np

class DragZoomImageHandler(Image):
    def __init__(self, rgb_img):
        super().__init__(allow_stretch=True)
        self.MIN_SIZE = 100
        self.img_size = rgb_img.shape[:2]
        self.current_zoom_size = np.array(self.img_size)
        self.current_zoom_pos = np.array([0,0])
        self.img_ratio = self.img_size[0]/(self.img_size[1]*1.0)
        self.set_texture(rgb_img)
        self.IS_DRAGGED = False
        self.touch_down_pos = None
        self.last_drag_pos = None
        self.callbacks = []


    def bind(self, callback_func):
        self.callbacks.append(callback_func)

    def collide_point(self, pos_x, pos_y):
        wpos_x, wpos_y = self.get_texture_pos()
        width,height = self.norm_image_size
        within_x = (pos_x > wpos_x and pos_x < (wpos_x+width))
        within_y = (pos_y > wpos_y and pos_y < (wpos_y+height))
        return (within_x and within_y)

    def get_texture_pos(self):
        outer_wx = self.width
        outer_wy = self.height
        outer_posx, outer_posy = self.pos
        center_x = outer_posx + outer_wx/2
        center_y = outer_posy + outer_wy/2
        inner_wx, inner_wy = self.norm_image_size
        bottom_left = (center_x-inner_wx/2, center_y-inner_wy/2)
        return bottom_left

    def get_cursor_pixel_pos(self, cursor_pos):
        cursor_pos_x, cursor_pos_y = cursor_pos
        tex_width, tex_height = self.get_norm_image_size()
        tex_pos_x, tex_pos_y = self.get_texture_pos()
        pixel_width = self.current_zoom_size[0]
        pixel_height = self.current_zoom_size[1]

        pixel_x = ((cursor_pos_x - tex_pos_x)/tex_width)*pixel_width
        pixel_y = ((cursor_pos_y - tex_pos_y)/tex_height)*pixel_height
        return (pixel_x, pixel_y)

    def get_global_cursor_pixel_pos(self, cursor_pos):
        rel_pix_x, rel_pix_y = self.get_cursor_pixel_pos(cursor_pos)
        return (rel_pix_x+self.current_zoom_pos[0], rel_pix_y+self.current_zoom_pos[1])


    def get_vecs_from_corners_to_cursor(self, cursor_pix_pos):
        pix_width = self.current_zoom_size[0]
        pix_height = self.current_zoom_size[1]
        cursor_pix_x, cursor_pix_y= cursor_pix_pos
        
        botleft_to_cursor = -np.array([-cursor_pix_x, -cursor_pix_y])
        topright_to_cursor = -np.array([pix_width-cursor_pix_x, pix_height-cursor_pix_y])
        return botleft_to_cursor, topright_to_cursor

    def correct_image_ratio(self, current_size):
        curr_width = current_size[0]
        curr_height = current_size[1]
        new_height = curr_width/self.img_ratio
        return np.array([curr_width, new_height])

    def is_above_min_size(self, current_size):
        curr_width = current_size[0]
        curr_height = current_size[1]
        return (curr_width>self.MIN_SIZE) and (curr_height>self.MIN_SIZE)

    def zoom_in_image(self, touch):
        if not self.is_above_min_size(self.current_zoom_size):
            return
        cursor_pix_pos = self.get_cursor_pixel_pos(touch.pos)
        botleft_to_cursor, topright_to_cursor = self.get_vecs_from_corners_to_cursor(cursor_pix_pos)
        scale_botleft = botleft_to_cursor*0.1
        scale_topright = topright_to_cursor*0.1

        self.current_zoom_size += (scale_topright.astype(np.int32)-scale_botleft.astype(np.int32))
        self.current_zoom_size = self.correct_image_ratio(self.current_zoom_size)
        self.current_zoom_pos += scale_botleft.astype(np.int32)

        self.texture = self.orig_texture.get_region(self.current_zoom_pos[0], self.current_zoom_pos[1], self.current_zoom_size[0], self.current_zoom_size[1])

    def zoom_out_image(self):
        orig_width = self.img_size[0]
        orig_height = self.img_size[1]
        current_width = self.current_zoom_size[0]
        current_height = self.current_zoom_size[1]
        if orig_width <= current_width and orig_height <= current_height:
            self.current_zoom_pos = np.array([0,0])
            return
        new_width = min(current_width*1.1, orig_width)
        new_height = min(current_height*1.1, orig_height)
        new_size = self.correct_image_ratio((new_width, new_height))
        new_width = new_size[0]
        new_height = new_size[1]
        current_pos_x = self.current_zoom_pos[0]
        current_pos_y = self.current_zoom_pos[1]
        delta_x = int((new_width-current_width)/2.0)
        delta_y = int((new_height-current_height)/2.0)
        if(current_pos_x+new_width+delta_x)>orig_width:
            delta_x += current_pos_x+new_width+delta_x-orig_width
        if(current_pos_y+new_height+delta_y)>orig_height:
            delta_y += current_pos_y+new_height+delta_y-orig_height
        new_x = max(0, current_pos_x-delta_x)
        new_y = max(0, current_pos_y-delta_y)
        self.current_zoom_pos = np.array([new_x, new_y])
        self.current_zoom_size = np.array([new_width, new_height])
        self.texture = self.orig_texture.get_region(int(new_x),int(new_y), int(new_width), int(new_height))

    def init_drag(self):
        Window.bind(mouse_pos=self.drag)
        while self.IS_DRAGGED:
            time.sleep(0.05)
            #print(self.get_cursor_pixel_pos())
        Window.unbind(mouse_pos=self.drag)
    
    def drag(self, instance, pos):
        pos = self.get_cursor_pixel_pos(pos)
        pos = np.array(pos)
        last_pos = np.array(self.last_drag_pos)
        self.last_drag_pos = pos
        pos_diff = last_pos-pos
        pos_diff_x = min(pos_diff[0], 100)
        pos_diff_y = min(pos_diff[1], 100)
        zoom_pos_x = self.current_zoom_pos[0]
        zoom_pos_y = self.current_zoom_pos[1]
        new_y = max(0, zoom_pos_y+pos_diff_y)
        new_x = max(0, zoom_pos_x+pos_diff_x)
        if (zoom_pos_x+pos_diff_x+self.current_zoom_size[0] > self.img_size[0]):
            print(zoom_pos_x+pos_diff_x+self.current_zoom_size[0])
            new_x = zoom_pos_x
        if (zoom_pos_y+pos_diff_y+self.current_zoom_size[1] > self.img_size[1]):
            new_y = zoom_pos_y
        self.current_zoom_pos = np.array([new_x, new_y])
        self.texture = self.orig_texture.get_region(int(new_x),int(new_y), self.current_zoom_size[0], self.current_zoom_size[1])


    def touch_down_equals_up_pos(self, pos1, pos2):
        return (pos1[0] == pos2[0] and pos1[1] == pos2[1])


    def on_touch_up(self, touch):
        if self.touch_down_pos is None:
            return
        self.IS_DRAGGED = False
        if self.collide_point(*touch.pos) and self.touch_down_equals_up_pos(touch.pos, self.touch_down_pos):
            if touch.is_mouse_scrolling:
                if touch.button == 'scrolldown':
                    self.zoom_in_image(touch)
                elif touch.button == 'scrollup':
                    self.zoom_out_image()
            else:
                self.call_callbacks(touch.pos)
        return super().on_touch_up(touch)

    def call_callbacks(self,pos):
        pix_pos = self.get_global_cursor_pixel_pos(pos)
        top_left_pix_pos = np.array([np.floor(self.img_size[1]-pix_pos[1]), np.floor(pix_pos[0]-0.25)]).astype(np.uint32)
        print(top_left_pix_pos)
        for cb in self.callbacks:
            cb(top_left_pix_pos)

    def on_touch_down(self, touch):
        self.last_drag_pos = self.get_cursor_pixel_pos(touch.pos)
        self.IS_DRAGGED = True
        self.touch_down_pos = touch.pos
        if self.collide_point(*touch.pos):
            thread = Thread(target=self.init_drag)
            thread.start()


    def set_texture(self, rgb_img):
        data = rgb_img.tobytes()
        # texture
        texture = Texture.create(size=(rgb_img.shape[1], rgb_img.shape[1]), colorfmt="rgb")
        texture.blit_buffer(data, bufferfmt="ubyte", colorfmt="rgb")
        texture.mag_filter = 'nearest'
        texture.min_filter = 'nearest'
        self.texture = texture
        self.orig_texture = texture
        if(rgb_img.shape[1] == self.img_size[0] and rgb_img.shape[0] == self.img_size[1]):
            self.texture = self.orig_texture.get_region(int(self.current_zoom_pos[0]), int(self.current_zoom_pos[1]), int(self.current_zoom_size[0]), int(self.current_zoom_size[1]))
        else:
            self.img_size = self.texture.size
            self.current_zoom_size = np.array(self.img_size)
            self.current_zoom_pos = np.array([0,0])

        



class DragZoomImage(AnchorLayout):
    def __init__(self, rgb_img=None,background_color=[0,0,0], **kwargs):
        super().__init__(anchor_x='center',  anchor_y='center', **kwargs)
        if rgb_img is None:
            rgb_img = np.zeros((512,512,3), dtype=np.uint8)
        rgb_img = cv2.flip(rgb_img, 0)
        self.image_handler = DragZoomImageHandler(rgb_img)
        self.add_widget(self.image_handler)

        self.bind(
            size=self._update_rect,
            pos=self._update_rect
        )

        with self.canvas.before:
            Color(background_color[0], background_color[1], background_color[2], 1)
            self.rect = Rectangle(
                size=self.size,
                pos=self.pos
            )

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size


    def bind_image_cb(self, callback):
        self.image_handler.bind(callback)

    def set_texture(self, rgb_img):
        rgb_img = cv2.flip(rgb_img, 0)
        self.image_handler.set_texture(rgb_img)

