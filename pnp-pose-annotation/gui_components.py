
from kivy.graphics.vertex_instructions import Rectangle
from kivy.graphics.context_instructions import Color


from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

class Sidebar(BoxLayout):
    def __init__(self):
        super().__init__(orientation='vertical', size_hint=(None, 1.0), width=400, padding=[0, 100, 0, 0])
        self.bind(
            size=self._update_rect,
            pos=self._update_rect
        )

        with self.canvas.before:
            Color(.1, .1, .1, 1)
            self.rect = Rectangle(
                size=self.size,
                pos=self.pos
            )

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size


class SidebarButton(Button):
    def __init__(self, text="", height=30, callback=None):
        super().__init__(size_hint=(1.0, None), text=text, height=height)
        if callback is not None:
            self.bind(callback)


class ColBoxLayout(BoxLayout):
    def __init__(self, bg_color, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            size=self._update_rect,
            pos=self._update_rect
        )

        print(bg_color)

        with self.canvas.before:
            Color(bg_color[0], bg_color[1], bg_color[2], 1)
            self.rect = Rectangle(
                size=self.size,
                pos=self.pos
            )

    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size






