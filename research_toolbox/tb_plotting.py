### plotting
import os
if "DISPLAY" not in os.environ:  # or os.environ["DISPLAY"] == ':0.0':
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import research_toolbox.tb_utils as tb_ut


class LinePlot:

    def __init__(self, title=None, xlabel=None, ylabel=None):
        self.data = []
        self.cfg = {'title': title, 'xlabel': xlabel, 'ylabel': ylabel}

    def add_line(self, xs, ys, label=None, err=None):
        d = {"xs": xs, "ys": ys, "label": label, "err": err}
        self.data.append(d)

    def plot(self, show=True, filepath=None):
        f = plt.figure()
        for d in self.data:
            plt.errorbar(d['xs'], d['ys'], yerr=d['err'], label=d['label'])

        plt.title(self.cfg['title'])
        plt.xlabel(self.cfg['xlabel'])
        plt.ylabel(self.cfg['ylabel'])
        plt.legend(loc='best')

        if filepath != None:
            f.savefig(filepath, bbox_inches='tight')
        if show:
            f.show()
        return f


# TODO: check this. this has not been tested.
class RunningLinePlot:

    def __init__(self):
        self.key2vs = {}

    def add(self, key, y, x=None):
        if key not in self.key2vs:
            d = {'ys': []}
            if x is not None:
                d["xs"] = []
            self.key2vs[key] = d

        d = self.key2vs[key]
        if x is not None:
            d["xs"].append(x)
        d["ys"].append(y)

    def plot(self, keys=None, show=True, filepath=None, title=None, xlabel=None, ylabel=None):
        f = plt.figure()

        if keys is None:
            keys = self.key2vs.keys()
        for key in keys:
            d = self.key2vs[key]
            if 'xs' in d:
                plt.plot(d["xs"], d["ys"], label=key)
            else:
                plt.plot(d["ys"], label=key)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(loc='best')

        if filepath != None:
            f.savefig(filepath, bbox_inches='tight')
        if show:
            f.show()
        return f