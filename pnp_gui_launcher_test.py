from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

class GUIMain(App):
    def build(self):
        main_window = BoxLayout(orientation='horizontal')
        return main_window


if __name__ == '__main__':
    GUIMain().run()
