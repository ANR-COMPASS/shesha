"""
To launch it :

    - locally :
        bokeh serve --show bokeh_display.py
    - as a server :
        bokeh serve --port 8081 --allow-websocket-origin hippo6.obspm.fr:8081 bokeh_roket.py
        then, open a web browser and connect to http://hippo6.obspm.fr:8081/bokeh_roket
"""
from widget_guardian import Bokeh_guardian
from bokeh.io import curdoc, output_file, show
import glob, os, atexit


def remove_files():
    files = glob.glob("/home/fferreira/public_html/roket_display*")
    for f in files:
        os.remove(f)


widget = Bokeh_guardian()
curdoc().clear()
curdoc().add_root(widget.tab)

atexit.register(remove_files)
