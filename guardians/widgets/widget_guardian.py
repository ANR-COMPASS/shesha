
from widget_roket import Bokeh_roket
from widget_gamora import Bokeh_gamora
from widget_groot import Bokeh_groot

from bokeh.models.widgets import Tabs


class Bokeh_guardian:
    """
    Class that defines a bokeh layout for all the guardians package
    Usage: see bokeh_roket.py which is the executable
    """

    def __init__(self):
        self.roket = Bokeh_roket()
        self.gamora = Bokeh_gamora()
        self.groot = Bokeh_groot()

        self.tab = Tabs(tabs=[self.roket.tab, self.gamora.tab, self.groot.tab])
