import sys
# sys.path.append("masks2ContoursScripts")

# This file is from https://matplotlib.org/3.1.1/gallery/widgets/lasso_selector_demo_sgskip.html.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector

class SelectFromCollection(object):
    '''
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.

    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to `alpha_other`.
    '''

    def __init__(self, ax, collection, ctype, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.ax = ax
        self.fig = self.ax.figure
        self.collection = collection
        self.alpha_other = alpha_other

        self.alpha_other = alpha_other
        self.is_subtract = True

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.run = True

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        # All points are initially selected.
        self.fc[:, -1] = 1 # 1 corresponds to "filled in" instead of translucent
        self.collection.set_facecolors(self.fc)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = np.array([], dtype=int)

        # Register the callback functions and store the associated connection ID's.
        self.press_cid = self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.release_cid = self.fig.canvas.mpl_connect("key_release_event", self.key_release)

        title = '''Select points of the {} you would like to remove.\n
            Hold down the left mouse button to use the lasso selector.\n
            Hold Shift to unremove points.\n
            Press Enter to confirm your selection.'''

        colors = {'LV endo': 'C0', 'LV epi': 'C1', 'RV free wall': 'C3'}
        ax.set_title(title.format(ctype))
        ax.plot([],[], 'o', label=ctype, color=colors[ctype])
        ax.legend(frameon=False)
        ax.set_aspect('equal')

    def onselect(self, verts):
        path = Path(verts)
        selected = np.nonzero(path.contains_points(self.xys))[0]

        # Subtract or add selections depending on whether shift is pressed or not.
        if self.is_subtract:
            self.ind = np.union1d(self.ind, selected)
            self.fc[self.ind, -1] = self.alpha_other
        else:
            self.ind = np.setdiff1d(self.ind, selected)
            self.fc[self.ind, -1] = 1

        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        self.canvas.mpl_disconnect(self.press_cid)
        self.canvas.mpl_disconnect(self.release_cid)

    def key_press(self, event):
        if event.key == "shift":
            self.is_subtract = False
        elif event.key == "enter":
            # Deconstruct the lasso selector.
            self.disconnect()
            self.ax.set_title("")
            self.canvas.draw()
            self.run = False
            plt.close('all')
            return self.ind

    def key_release(self, event):
        if event.key == "shift":
            self.is_subtract = True