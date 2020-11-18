import tkinter as tk
from enum import Enum
from typing import List, Dict
import numpy as np
from tensorflow.python.keras.models import load_model
from AIForecast.access import WeatherAccess
from AIForecast import utils
from AIForecast.RNN import _RNN

BACKGROUND_COLOR = '#525453'
"""
Default background color for most UI elements.
"""

FOREGROUND_COLOR = 'white'

H1_FONT = 100

BUTTON_BORDER_WIDTH = 1
BUTTON_BACKGROUND = 'white'
BUTTON_ACTIVE_BACKGROUND = 'gray'
BUTTON_FOREGROUND = 'black'

_ALIGN_X = 10
_ALIGN_Y = 10


class Menus(Enum):
    """
    Menus is an enum containing named menus that exist in the application.

    Author: Alexander Cherry
    """

    MAIN_MENU = 0
    TEST_MENU = 1
    TRAIN_MENU = 2


class Drawable:

    def init_ui(self):
        """
        Init_ui is an abstract function.
        Use init_ui to instantiate ui elements before drawing them.

        Author: Alexander Cherry
        """
        pass

    def draw(self):
        """
        Draw is an abstract function.
        Use draw to show elements of the inheriting class.
        It is recommended to invoke the super draw method before drawing additional elements.

        Author: Alexander Cherry
        """
        pass

    def hide(self):
        """
        Hide is an abstract function.
        Use hide to destroy all elements that were added to the screen by the draw method.
        It is recommended to invoke the super hide method to ensure all nested elements are remove.

        Author: Alexander Cherry
        """
        pass


class NavBar(Drawable):
    NAV_COLOR = 'gray'
    NAV_HEIGHT = 40

    def __init__(self, parent: tk.Frame):
        self.parent = parent
        self.nav_buttons: List[tk.Button] = []
        self.nav_frame = None

    def init_ui(self):
        self.nav_frame = tk.Frame(self.parent, bg=self.NAV_COLOR)
        self.nav_buttons.append(
            tk.Button(
                self.nav_frame,
                text="Main",
                borderwidth=BUTTON_BORDER_WIDTH,
                bg=BUTTON_BACKGROUND,
                fg=BUTTON_FOREGROUND,
                command=lambda: AppWindow.display_screen(Menus.MAIN_MENU)
            )
        )
        self.nav_buttons.append(
            tk.Button(
                self.nav_frame,
                text="Test",
                borderwidth=BUTTON_BORDER_WIDTH,
                bg=BUTTON_BACKGROUND,
                fg=BUTTON_FOREGROUND,
                command=lambda: AppWindow.display_screen(Menus.TEST_MENU)
            )
        )
        self.nav_buttons.append(
            tk.Button(
                self.nav_frame,
                text="Train",
                borderwidth=BUTTON_BORDER_WIDTH,
                bg=BUTTON_BACKGROUND,
                fg=BUTTON_FOREGROUND,
                command=lambda: AppWindow.display_screen(Menus.TRAIN_MENU)
            )
        )

    def draw(self):
        self.nav_frame.place(x=0, y=0, relwidth=1, height=self.NAV_HEIGHT)
        i = 0
        for button in self.nav_buttons:
            button.place(x=10 + 100 * i + 5 * i, rely=0.2, relheight=0.55, width=100)
            i += 1

    def hide(self):
        self.nav_frame.destroy()
        for button in self.nav_buttons:
            button.destroy()
        self.nav_buttons.clear()


class Menu(Drawable):
    """
    A standard Menu has a Nav bar and a body.
    The body of the Menu is initially blank.
    Classes extending Menu are tasked with filling the body
    with content.

    Author: Alexander Cherry
    """

    def __init__(self, app_frame: tk.Frame):
        self.container = app_frame
        self.body_height = 0
        self.nav_bar = NavBar(self.container)
        self.body = None

    def init_ui(self):
        self.nav_bar.init_ui()
        self.body = tk.Frame(self.container, bg=BACKGROUND_COLOR)

    def draw(self):
        self.nav_bar.draw()
        self.body_height = AppWindow.current_height - NavBar.NAV_HEIGHT
        self.body.place(x=0, y=NavBar.NAV_HEIGHT, relwidth=1, height=self.body_height)

    def hide(self):
        self.nav_bar.hide()
        self.body.destroy()


class MainMenu(Menu):
    LABEL_TEXT = """
    Training AI to Predict Temperatures
    by
    Alexander Cherry, Anthony Ernst, and Marcus Kline

    Previously worked on by:
    Jason, Swatt, and Tiger
    """

    def __init__(self, app_frame: tk.Frame):
        Menu.__init__(self, app_frame)
        self.label = None

    def init_ui(self):
        Menu.init_ui(self)
        self.label = tk.Label(self.body, text=self.LABEL_TEXT, bg=BACKGROUND_COLOR, fg=FOREGROUND_COLOR)

    def draw(self):
        Menu.draw(self)
        self.label.place(relx=0, rely=0, relwidth=1, relheight=0.5)

    def hide(self):
        Menu.hide(self)
        self.label.destroy()


class TestMenu(Menu):

    def __init__(self, app_frame: tk.Frame):
        Menu.__init__(self, app_frame)
        self.input_frame = None  # Type: tk.Frame
        self.left_pane = None  # Type: tk.Frame
        self.source_select_label = None  # Type: tk.Label
        self.source_current_radio = None  # Type: tk.Radiobutton
        self.source_manual_radio = None  # Type: tk.Radiobutton
        self.source_desc_label = None  # Type: tk.Label
        self.radio_group = tk.IntVar()
        self.right_pane = None  # Type: tk.Frame
        self.surrounding_cities_label = None  # Type: tk.Label
        self.surrounding_cities_listbox = None  # Type: tk.Listbox
        self.target_cities_label = None  # Type: tk.Label
        self.target_cities_listbox = None  # Type: tk.Listbox
        self.enter_button = None  # Type: tk.Button
        self.output_frame = None  # Type: tk.Frame
        self.button_desc_label = None  # Type: tk.Label
        self.output_text = None  # Type: tk.Text
        self.run_button = None  # Type: tk.Button

    def init_ui(self):
        Menu.init_ui(self)
        self.input_frame = tk.Frame(self.body, bg=BACKGROUND_COLOR)

        # Add all of the input elements
        self.left_pane = tk.Frame(self.input_frame, bg=BACKGROUND_COLOR)
        self.source_select_label = tk.Label(
            self.left_pane,
            text="Choose Input Source:",
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            font=H1_FONT
        )
        self.source_current_radio = tk.Radiobutton(
            self.left_pane,
            text="Current",
            variable=self.radio_group,
            value=1,
            borderwidth=5,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            selectcolor='gray',
            command=self._run_radio_1
        )
        self.source_manual_radio = tk.Radiobutton(
            self.left_pane,
            text="Manual",
            variable=self.radio_group,
            value=2,
            borderwidth=5,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            selectcolor='gray',
            command=self._run_radio_2
        )
        self.source_desc_label = tk.Label(
            self.left_pane,
            text="temporary",
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR
        )
        self.enter_button = tk.Button(
            self.left_pane,
            text="Enter Data",
            bg=BUTTON_BACKGROUND,
            fg=BUTTON_FOREGROUND,
            command=self._enter_data  # Todo: add functionality for this button
        )

        self.right_pane = tk.Frame(self.input_frame, bg=BACKGROUND_COLOR)
        self.surrounding_cities_label = tk.Label(
            self.right_pane,
            text="Choose surrounding cities from the list:",
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            font=H1_FONT
        )
        self.surrounding_cities_listbox = tk.Listbox(self.right_pane)
        self.target_cities_label = tk.Label(
            self.right_pane,
            text="Choose a target city from the list:",
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            font=H1_FONT
        )
        self.target_cities_listbox = tk.Listbox(self.right_pane)

        self.output_frame = tk.Frame(self.body, bg=BACKGROUND_COLOR)
        self.button_desc_label = tk.Label(
            self.output_frame,
            text="Press the button below to generate a prediction:",
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            font=H1_FONT
        )
        self.run_button = tk.Button(
            self.output_frame,
            text="Run",
            borderwidth=BUTTON_BORDER_WIDTH,
            command=lambda: self._run_test
        )
        self.output_text = tk.Text(self.output_frame)

        self.source_current_radio.select()
        self.source_current_radio.invoke()

    def draw(self):
        Menu.draw(self)
        self.input_frame.place(relx=0, rely=0, relwidth=1, relheight=0.5)
        self.left_pane.place(relx=0, rely=0, relwidth=0.3, relheight=1)
        self.source_select_label.place(x=_ALIGN_X, y=_ALIGN_Y)
        label_height = self.source_select_label.winfo_reqheight()
        radio_width = self.source_current_radio.winfo_reqwidth()
        self.source_current_radio.place(x=_ALIGN_X, y=_ALIGN_Y + label_height)
        self.source_manual_radio.place(x=_ALIGN_X + radio_width + 5, y=_ALIGN_Y + label_height)
        self.source_desc_label.place(x=_ALIGN_X, y=_ALIGN_Y * 7)
        self.enter_button.place(x=_ALIGN_X + 5, y=_ALIGN_Y * 11)

        # Todo: Replace selection lists with a textbox.
        # Content in the right pane will be replaced with a textbox in the future to work with pyowm instead of
        # being a selection list.
        self.right_pane.place(relx=0.3, rely=0, relwidth=0.7, relheight=1)
        self.surrounding_cities_label.place(x=_ALIGN_X, y=_ALIGN_Y)
        city_label_width = self.surrounding_cities_label.winfo_reqwidth()
        self.surrounding_cities_listbox.place(
            x=_ALIGN_X + 5,
            y=_ALIGN_Y + label_height,
            width=city_label_width
        )
        self.target_cities_label.place(x=city_label_width + 30, y=_ALIGN_Y)
        target_label_width = self.target_cities_label.winfo_reqwidth()
        self.target_cities_listbox.place(
            x=city_label_width + 35,
            y=_ALIGN_Y + label_height,
            width=target_label_width
        )
        # Draw Input frame stuffs here:

        self.output_frame.place(relx=0, rely=0.5, relwidth=1, relheight=0.5)
        self.button_desc_label.place(x=_ALIGN_X, y=_ALIGN_Y)
        self.run_button.place(x=_ALIGN_X, y=_ALIGN_Y + label_height + 5, width=80, height=40)
        self.output_text.place(x=_ALIGN_X, y=_ALIGN_Y + label_height + 50, relwidth=0.5, relheight=0.5)
        # Draw Output frame stuffs here:

    def hide(self):
        Menu.hide(self)
        self.input_frame.destroy()
        self.left_pane.destroy()
        self.source_select_label.destroy()
        self.source_current_radio.destroy()
        self.source_manual_radio.destroy()
        self.source_desc_label.destroy()
        self.right_pane.destroy()
        self.surrounding_cities_label.destroy()
        self.surrounding_cities_listbox.destroy()
        self.target_cities_label.destroy()
        self.target_cities_listbox.destroy()
        self.enter_button.destroy()
        self.output_frame.destroy()
        self.button_desc_label.destroy()
        self.output_text.destroy()
        self.run_button.destroy()

    def _run_radio_1(self):
        """
        run_radio_1 is the command routine for source_current_radio.
        """
        self.source_desc_label.config(text="Use current data to make a prediction.")
        self.run_button.config(state='normal')
        self.surrounding_cities_listbox.config(state='disabled')
        self.target_cities_listbox.config(state='disabled')
        self.enter_button.config(state='disabled')

    def _run_radio_2(self):
        """
        run_radio_2 is the command routine for source_manual_radio.
        """
        self.source_desc_label.config(text="Manually input data to make a prediction.")
        self.run_button.config(state='normal')
        self.surrounding_cities_listbox.config(state='normal')
        self.target_cities_listbox.config(state='normal')
        self.enter_button.config(state='normal')

    def on_select(self, event):
        """
        Todo: This may not be needed
        """
        pass

    def _populate_surrounding_cities(self):
        """
        This will be replaced by a more robust system.
        Todo: Remove this all together and add auto city detection.
        """
        pass

    def _populate_target_cities(self):
        """
        This will be replaced by a more robust system.
        Todo: Create a textbox instead of a selection box for this
        """
        pass

    def _enter_data(self):
        """
        This function was not implemented yet.
        Todo: Implement this? Is this necessary?
        """
        pass

    def _run_test(self):
        """
        Todo: Implement the testing strategy.
        """
        model = load_model(utils.path_utils.get_model_path() + "/model-50.hdf5")
        test_data = np.zeros((1, 1, 3))
        prediction = model.predict(test_data)
        utils.log(self.__name__).debug("Loading Weather Model!")


class TrainMenu(Menu):

    def __init__(self, app_frame: tk.Frame):
        Menu.__init__(self, app_frame)
        self.select_label = None
        self.historic_radio = None
        self.user_radio = None
        self.radio_group = tk.IntVar()
        self.run_training = None
        self.output_text = None

    def init_ui(self):
        Menu.init_ui(self)
        self.select_label = tk.Label(
            self.body,
            text="Choose Input Source:",
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            font=H1_FONT
        )
        self.historic_radio = tk.Radiobutton(
            self.body,
            text="Historic Data",
            variable=self.radio_group,
            value=1,
            borderwidth=5,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            selectcolor='gray',
            command=lambda: print("test")
        )
        self.user_radio = tk.Radiobutton(
            self.body,
            text="User Data",
            variable=self.radio_group,
            value=2,
            borderwidth=5,
            bg=BACKGROUND_COLOR,
            fg=FOREGROUND_COLOR,
            selectcolor='gray',
            command=lambda: print("test2")
        )
        self.run_training = tk.Button(
            self.body,
            text="Run",
            borderwidth=BUTTON_BORDER_WIDTH,
            command=lambda: self._run_train()
        )
        self.output_text = tk.Text(self.body)

    def draw(self):
        Menu.draw(self)
        self.select_label.place(x=_ALIGN_X, y=_ALIGN_Y)
        label_height = self.select_label.winfo_reqheight()
        radio_width = self.select_label.winfo_reqwidth()
        self.historic_radio.place(x=_ALIGN_X, y=_ALIGN_Y + label_height)
        self.user_radio.place(x=_ALIGN_X + radio_width + 5, y=_ALIGN_Y + label_height)
        self.run_training.place(x=_ALIGN_X, y=_ALIGN_Y + label_height + 50)
        self.output_text.place(x=_ALIGN_X, y=_ALIGN_Y + label_height + 80, relwidth=0.8, relheight=0.5)

    def hide(self):
        Menu.hide(self)
        self.select_label.destroy()
        self.historic_radio.destroy()
        self.user_radio.destroy()
        self.run_training.destroy()
        self.output_text.destroy()

    def _run_train(self):
        """
        Trains historical weather data through a Recurring Neural Network (RNN)
        :return: NULL
        Author: Marcus Kline
        """
        rnn_object = _RNN.RNN()
        self.output_text.insert(rnn_object.run_rnn())


class OptionsMenu(Menu):
    pass


class AppWindow:
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 600

    _WINDOW_DIMENSIONS = {
        "relx": 0,
        "rely": 0,
        "relwidth": 1.0,
        "relheight": 1.0
    }

    current_width = WINDOW_WIDTH
    current_height = WINDOW_HEIGHT

    menu_list: Dict = {}
    current_menu = None

    def __init__(self, root: tk.Tk):
        self.window = tk.Canvas(root, width=self.WINDOW_WIDTH, height=self.WINDOW_HEIGHT, highlightthickness=0)
        self.frame = tk.Frame(root, bg=BACKGROUND_COLOR)
        self.window.pack(fill='both', expand=True)
        self.frame.place(**self._WINDOW_DIMENSIONS)

        self.window.bind("<Configure>", self.on_resize)

    @staticmethod
    def on_resize(event):
        AppWindow.current_width, AppWindow.current_height = event.width, event.height
        if AppWindow.current_menu is not None:
            AppWindow.current_menu.draw()

    @staticmethod
    def register_menu(screen: Menu, name: Menus):
        AppWindow.menu_list[name] = screen

    @staticmethod
    def display_screen(screen: Menus):
        utils.log(__name__).debug("Opening " + str(screen) + "!")
        if AppWindow.current_menu is not None:
            AppWindow.current_menu.hide()
        AppWindow.current_menu = AppWindow.menu_list[screen]
        AppWindow.current_menu.init_ui()
        AppWindow.current_menu.draw()
