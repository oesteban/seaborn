from __future__ import division
from itertools import product
from distutils.version import LooseVersion
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from six import string_types

from . import utils
from .palettes import color_palette, light_palette


class ConditionalJointGrid(object):

    """Grid for drawing a bivariate plot with marginal univariate plots."""

    def __init__(self, x, y, data=None, hue=None, hue_order=None, size=6,
                 ratio=5, space=.2, dropna=True, xlim=None, ylim=None,
                 inline_labels=False):
        """Set up the grid of subplots.

        Parameters
        ----------
        x, y : strings or vectors
            Data or names of variables in `data`.
        data : DataFrame, optional
            DataFrame when `x` and `y` are variable names.
        hue : string (variable name), optional
            Variable in ``data`` to map plot aspects to different colors.
        size : numeric
            Size of the figure (it will be square).
        ratio : numeric
            Ratio of joint axes size to marginal axes height.
        space : numeric, optional
            Space between the joint and marginal axes
        dropna : bool, optional
            If True, remove observations that are missing from `x` and `y`.
        {x, y}lim : two-tuples, optional
            Axis limits to set before plotting.

        See Also
        --------
        jointplot : Inteface for drawing bivariate plots with several different
                    default plot kinds.

        """
        # Set up the subplot grid
        f = plt.figure(figsize=(size, size))
        gs = plt.GridSpec(ratio + 1, ratio + 1)

        ax_joint = f.add_subplot(gs[1:, :-1])
        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
        ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

        self.fig = f
        self.ax_joint = ax_joint
        self.ax_marg_x = ax_marg_x
        self.ax_marg_y = ax_marg_y
        self.inline_labels = inline_labels

        # Turn off tick visibility for the measure axis on the marginal plots
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)

        # Turn off the ticks on the density axis for the marginal plots
        plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
        plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
        plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
        plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
        plt.setp(ax_marg_x.get_yticklabels(), visible=False)
        plt.setp(ax_marg_y.get_xticklabels(), visible=False)
        ax_marg_x.yaxis.grid(False)
        ax_marg_y.xaxis.grid(False)

        # Possibly drop NA
        if dropna:
            not_na = pd.notnull(data[x]) & pd.notnull(data[y])
            data = data[not_na]

        # Possibly extract the variables from a DataFrame
        if data is not None:
            if x in data:
                x = data[x]
            if y in data:
                y = data[y]

        # Find the names of the variables
        if hasattr(x, "name"):
            xlabel = x.name
            ax_joint.set_xlabel(xlabel)
        if hasattr(y, "name"):
            ylabel = y.name
            ax_joint.set_ylabel(ylabel)

        # Convert the x and y data to arrays for plotting
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        if xlim is not None:
            ax_joint.set_xlim(xlim)
        if ylim is not None:
            ax_joint.set_ylim(ylim)

        # Sort out the hue variable
        self._hue_var = hue
        if hue is None:
            self.hue_names = [None]
            self.hue_vals = pd.Series(["_nolegend_"] * len(data),
                                      index=data.index)
        else:
            if hue_order is None:
                hue_names = np.atleast_1d(
                    np.unique(np.sort(data[hue]))).tolist()
            else:
                hue_names = hue_order

            self.hue_names = hue_names
            self.hue_vals = data[hue]

        # Additional dict of kwarg -> list of values for mapping the hue var
        # self.hue_kws = hue_kws if hue_kws is not None else {}
        self.palette = color_palette("husl", n_colors=len(self.hue_names))

        # Make the grid look nice
        utils.despine(f)
        utils.despine(ax=ax_marg_x, left=True)
        utils.despine(ax=ax_marg_y, bottom=True)
        f.tight_layout()
        f.subplots_adjust(hspace=space, wspace=space)

    def plot(self, joint_func, marginal_func, annot_func=None):
        """Shortcut to draw the full plot.

        Use `plot_joint` and `plot_marginals` directly for more control.

        Parameters
        ----------
        joint_func, marginal_func: callables
            Functions to draw the bivariate and univariate plots.

        Returns
        -------
        self : JointGrid instance
            Returns `self`.

        """
        self.plot_marginals(marginal_func)
        self.plot_joint(joint_func)
        if annot_func is not None:
            self.annotate(annot_func)
        return self

    def plot_joint(self, func, **kwargs):
        """Draw a bivariate plot of `x` and `y`.

        Parameters
        ----------
        func : plotting callable
            This must take two 1d arrays of data as the first two
            positional arguments, and it must plot on the "current" axes.
        kwargs : key, value mappings
            Keyword argument are passed to the plotting function.

        Returns
        -------
        self : JointGrid instance
            Returns `self`.

        """
        from .distributions import kdeplot
        from matplotlib import patches as mpatches

        plt.sca(self.ax_joint)

        isscatter = (func == plt.scatter)
        colorkw = 'cmap'

        patches = []

        if func == plt.scatter:
            kwargs['edgecolor'] = 'white'
            func = getattr(self.ax_joint, 'scatter')
            colorkw = 'c'

        for k, hue in enumerate(self.hue_names):
            if hue is not None:
                kwargs['label'] = hue
                x = self.x[np.where(self.hue_vals == hue)]
                y = self.y[np.where(self.hue_vals == hue)]
            else:
                x = self.x
                y = self.y

            thiscolor = self.palette[k]

            if colorkw == 'c':
                kwargs['c'] = thiscolor
            elif colorkw == 'cmap':
                kwargs['cmap'] = light_palette(
                    thiscolor, as_cmap=True)
            func(x, y, **kwargs)

            if hue is not None and self.inline_labels:
                mu = (np.median(x), np.median(y))
                self.ax_joint.annotate(
                    hue, xy=(mu[0], mu[1]), xytext=(30, 20),
                    textcoords='offset points', size=30, va='center',
                    color='w',
                    bbox=dict(boxstyle="round", fc=thiscolor, ec='none',
                              alpha=0.7, color='w')
                )
            elif hue is not None and not self.inline_labels:
                patches.append(mpatches.Patch(color=thiscolor, label=hue))

        if len(patches) > 0:
            self.ax_joint.legend(handles=patches)

        return self

    def plot_marginals(self, func, **kwargs):
        """Draw univariate plots for `x` and `y` separately.

        Parameters
        ----------
        func : plotting callable
            This must take a 1d array of data as the first positional
            argument, it must plot on the "current" axes, and it must
            accept a "vertical" keyword argument to orient the measure
            dimension of the plot vertically.
        kwargs : key, value mappings
            Keyword argument are passed to the plotting function.

        Returns
        -------
        self : JointGrid instance
            Returns `self`.

        """
        kwargs["vertical"] = False
        plt.sca(self.ax_marg_x)

        for k, hue in enumerate(self.hue_names):
            if hue is not None:
                kwargs['label'] = hue
                x = self.x[np.where(self.hue_vals == hue)]
            else:
                x = self.x
            func(x, color=self.palette[k], **kwargs)

        kwargs["vertical"] = True
        plt.sca(self.ax_marg_y)

        for k, hue in enumerate(self.hue_names):
            if hue is not None:
                kwargs['label'] = hue
                y = self.y[np.where(self.hue_vals == hue)]
            else:
                y = self.y
            func(y, color=self.palette[k], **kwargs)

        try:
            self.ax_marg_x.legend_.remove()
        except AttributeError:
            pass

        try:
            self.ax_marg_y.legend_.remove()
        except AttributeError:
            pass

        return self

    def annotate(self, func, template=None, stat=None, loc="best", **kwargs):
        """Annotate the plot with a statistic about the relationship.

        Parameters
        ----------
        func : callable
            Statistical function that maps the x, y vectors either to (val, p)
            or to val.
        template : string format template, optional
            The template must have the format keys "stat" and "val";
            if `func` returns a p value, it should also have the key "p".
        stat : string, optional
            Name to use for the statistic in the annotation, by default it
            uses the name of `func`.
        loc : string or int, optional
            Matplotlib legend location code; used to place the annotation.
        kwargs : key, value mappings
            Other keyword arguments are passed to `ax.legend`, which formats
            the annotation.

        Returns
        -------
        self : JointGrid instance.
            Returns `self`.

        """
        default_template = "{stat} = {val:.2g}; p = {p:.2g}"

        # Call the function and determine the form of the return value(s)
        out = func(self.x, self.y)
        try:
            val, p = out
        except TypeError:
            val, p = out, None
            default_template, _ = default_template.split(";")

        # Set the default template
        if template is None:
            template = default_template

        # Default to name of the function
        if stat is None:
            stat = func.__name__

        # Format the annotation
        if p is None:
            annotation = template.format(stat=stat, val=val)
        else:
            annotation = template.format(stat=stat, val=val, p=p)

        # Draw an invisible plot and use the legend to draw the annotation
        # This is a bit of a hack, but `loc=best` works nicely and is not
        # easily abstracted.
        phantom, = self.ax_joint.plot(self.x, self.y, linestyle="", alpha=0)
        self.ax_joint.legend([phantom], [annotation], loc=loc, **kwargs)
        phantom.remove()

        return self

    def set_axis_labels(self, xlabel="", ylabel="", **kwargs):
        """Set the axis labels on the bivariate axes.

        Parameters
        ----------
        xlabel, ylabel : strings
            Label names for the x and y variables.
        kwargs : key, value mappings
            Other keyword arguments are passed to the set_xlabel or
            set_ylabel.

        Returns
        -------
        self : JointGrid instance
            returns `self`

        """
        self.ax_joint.set_xlabel(xlabel, **kwargs)
        self.ax_joint.set_ylabel(ylabel, **kwargs)
        return self

    def savefig(self, *args, **kwargs):
        """Wrap figure.savefig defaulting to tight bounding box."""
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(*args, **kwargs)
