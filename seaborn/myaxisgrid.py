from __future__ import division
from itertools import product
from distutils.version import LooseVersion
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgs

from six import string_types

from . import utils
from .palettes import color_palette, light_palette


class ConditionalJointGrid(object):

    """Grid for drawing a bivariate plot with marginal univariate plots."""

    def __init__(self, x, y, data=None, hue=None, hue_order=None, size=6,
                 ratio=5, space=.2, dropna=True, xlim=None, ylim=None,
                 inline_labels=False, splitgrid=True):
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

        self.splitgrid = splitgrid
        self.inline_labels = inline_labels
        self.ratio = ratio
        gwidth = ratio + 1

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

        self.n_splits = 1

        nrows = 0
        ncols = 0

        if splitgrid and hue is not None:
            self.n_splits = len(hue_names) + 1

            nrows = 2
            ncols = int(np.ceil(len(hue_names) / float(nrows)))

        # Set up the subplot grid
        f = plt.figure(figsize=(size * (1 + ncols * 0.5), size))

        gs0 = mgs.GridSpec(1, 2, width_ratios=[1, ncols / 2.0])
        gs1 = mgs.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0],
                                          width_ratios=[ratio, 1],
                                          height_ratios=[1, ratio])

        gs2 = mgs.GridSpecFromSubplotSpec(nrows * 2, ncols * 2,
                                          subplot_spec=gs0[1],
                                          width_ratios=[ratio, 1] * ncols,
                                          height_ratios=[1, ratio] * nrows)

        ax_joint = [f.add_subplot(gs1[1, 0])]
        ax_marg_x = [f.add_subplot(gs1[0, 0], sharex=ax_joint[-1])]
        ax_marg_y = [f.add_subplot(gs1[1, 1], sharey=ax_joint[-1])]

        hueid = 0
        for row in range(nrows):
            for col in range(ncols):
                ax_joint.append(f.add_subplot(gs2[(row * 2 + 1), col * 2]))
                ax_marg_x.append(f.add_subplot(
                    gs2[row * 2, col * 2], sharex=ax_joint[-1]))
                ax_marg_y.append(f.add_subplot(
                    gs2[(row * 2 + 1), (col * 2 + 1)], sharey=ax_joint[-1]))
                hueid += 1

                if hueid >= len(hue_names):
                    break

        self.fig = f
        self.ax_joint = ax_joint
        self.ax_marg_x = ax_marg_x
        self.ax_marg_y = ax_marg_y

        # Find the names of the variables
        if hasattr(x, "name"):
            xlabel = x.name
            ax_joint[0].set_xlabel(xlabel)
        if hasattr(y, "name"):
            ylabel = y.name
            ax_joint[0].set_ylabel(ylabel)

        # Convert the x and y data to arrays for plotting
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        if xlim is not None:
            for axj in ax_joint:
                axj.set_xlim(xlim)

        if ylim is not None:
            for axj in ax_joint:
                axj.set_ylim(ylim)

        # Additional dict of kwarg -> list of values for mapping the hue var
        # self.hue_kws = hue_kws if hue_kws is not None else {}
        self.palette = color_palette("husl", n_colors=len(self.hue_names))

        # Make the grid look nice
        utils.despine(f)

        for ax_mx, ax_my in zip(ax_marg_x, ax_marg_y):
            # Turn off tick visibility for the measure axis
            # on the marginal plots
            plt.setp(ax_mx.get_xticklabels(), visible=False)
            plt.setp(ax_my.get_yticklabels(), visible=False)

            # Turn off the ticks on the density axis for the marginal plots
            plt.setp(ax_mx.yaxis.get_majorticklines(), visible=False)
            plt.setp(ax_mx.yaxis.get_minorticklines(), visible=False)
            plt.setp(ax_my.xaxis.get_majorticklines(), visible=False)
            plt.setp(ax_my.xaxis.get_minorticklines(), visible=False)
            plt.setp(ax_mx.get_yticklabels(), visible=False)
            plt.setp(ax_my.get_xticklabels(), visible=False)
            ax_mx.yaxis.grid(False)
            ax_my.xaxis.grid(False)
            utils.despine(ax=ax_mx, left=True)
            utils.despine(ax=ax_my, bottom=True)

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
        self : ConditionalJointGrid instance
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
        self : ConditionalJointGrid instance
            Returns `self`.

        """
        from .distributions import kdeplot
        from matplotlib import patches as mpatches

        colorkw = 'cmap'

        patches = []
        thecolors = []

        if func == plt.scatter:
            kwargs['edgecolor'] = 'white'
            func = getattr(self.ax_joint[0], 'scatter')
            colorkw = 'c'

        for k, hue in enumerate(self.hue_names):
            if hue is not None:
                kwargs['label'] = hue
                x = self.x[np.where(self.hue_vals == hue)]
                y = self.y[np.where(self.hue_vals == hue)]
            else:
                x = self.x
                y = self.y

            for p in range(self.n_splits):
                plt.sca(self.ax_joint[p])

                if p == 0 or p == k + 1:
                    thiscolor = self.palette[k]
                else:
                    thiscolor = 'darkgray'

                if colorkw == 'c':
                    kwargs['c'] = thiscolor
                elif colorkw == 'cmap':
                    kwargs['cmap'] = light_palette(
                        thiscolor, as_cmap=True)

                if p == 0:
                    func(x, y, **kwargs)
                elif p != k + 1:
                    subkwargs = kwargs.copy()
                    subkwargs['c'] = thiscolor
                    subkwargs['edgecolor'] = 'white'
                    subkwargs['linewidths'] = .25
                    self.ax_joint[p].scatter(x, y, **subkwargs)

                kwargs.pop('c', None)
                kwargs.pop('cmap', None)

                if p == 0:
                    if hue is not None and self.inline_labels:
                        mu = (np.median(x), np.median(y))
                        self.ax_joint[p].annotate(
                            hue, xy=(mu[0], mu[1]), xytext=(30, 20),
                            textcoords='offset points', size=30, va='center',
                            color='w',
                            bbox=dict(boxstyle="round", fc=self.palette[k],
                                      ec='none', alpha=0.7, color='w')
                        )
                    elif hue is not None and not self.inline_labels:
                        patches.append(mpatches.Patch(color=self.palette[k],
                                                      label=hue))
        if self.n_splits > 1:
            for k, hue in enumerate(self.hue_names):
                kwargs['label'] = hue
                x = self.x[np.where(self.hue_vals == hue)]
                y = self.y[np.where(self.hue_vals == hue)]

                kwargs['c'] = self.palette[k]
                kwargs['edgecolor'] = 'white'
                kwargs['linewidths'] = .25
                self.ax_joint[k + 1].scatter(x, y, **kwargs)

                if self.inline_labels:
                    mu = (0.75 * self.ax_joint[k + 1].get_xlim()[0],
                          1.1 * self.ax_joint[k + 1].get_ylim()[0])
                    self.ax_joint[k + 1].annotate(
                        hue, xy=(mu[0], mu[1]), xytext=(30, 20),
                        textcoords='offset points', size=20, va='center',
                        color='w',
                        bbox=dict(boxstyle="round", fc=self.palette[k],
                                  ec='none', alpha=0.7, color='w')
                    )
                elif hue is not None and not self.inline_labels:
                    patches.append(mpatches.Patch(color=self.palette[k],
                                                  label=hue))

        if len(patches) > 0:
            self.ax_joint[0].legend(handles=patches)

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
        self : ConditionalJointGrid instance
            Returns `self`.

        """
        kwargs["vertical"] = False

        for k, hue in enumerate(self.hue_names):
            if hue is not None:
                kwargs['label'] = hue
                x = self.x[np.where(self.hue_vals == hue)]
            else:
                x = self.x

            for p in range(self.n_splits):
                if p == 0 or p == k + 1:
                    c = self.palette[k]
                else:
                    c = 'darkgray'
                plt.sca(self.ax_marg_x[p])
                func(x, color=c, **kwargs)

                try:
                    self.ax_marg_x[p].legend_.remove()
                except AttributeError:
                    pass

        kwargs["vertical"] = True
        plt.sca(self.ax_marg_y[0])

        for k, hue in enumerate(self.hue_names):
            if hue is not None:
                kwargs['label'] = hue
                y = self.y[np.where(self.hue_vals == hue)]
            else:
                y = self.y

            for p in range(self.n_splits):
                if p == 0 or p == k + 1:
                    c = self.palette[k]
                else:
                    c = 'darkgray'
                plt.sca(self.ax_marg_y[p])
                func(y, color=c, **kwargs)

                try:
                    self.ax_marg_y[p].legend_.remove()
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
        self : ConditionalJointGrid instance.
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
        phantom, = self.ax_joint[0].plot(self.x, self.y, linestyle="", alpha=0)
        self.ax_joint[0].legend([phantom], [annotation], loc=loc, **kwargs)
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
        self : ConditionalJointGrid instance
            returns `self`

        """
        self.ax_joint[0].set_xlabel(xlabel, **kwargs)

        for ax in self.ax_joint:
            ax.set_ylabel(ylabel, **kwargs)
        return self

    def savefig(self, *args, **kwargs):
        """Wrap figure.savefig defaulting to tight bounding box."""
        kwargs.setdefault("bbox_inches", "tight")
        self.fig.savefig(*args, **kwargs)
