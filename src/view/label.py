from typing import Optional

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter
from torch import nn

import mypython.plotutil as mpu
import tool.util
from mypython.terminal import Color


class Label:
    def __init__(self, domain: Optional[str]) -> None:
        self.domain = domain

        self.latent_0 = r"latent element (1)"
        self.latent_1 = r"latent element (2)"
        self.latent_2 = r"latent element (3)"

        # TODO:
        self.latent_0_range = None
        self.latent_1_range = None

        if type(domain) == str:
            if domain == "reacher2d":
                self.physical_0 = r"physical angle ($\theta_1$)"
                self.physical_1 = r"physical angle ($\theta_2$)"
                self.physical_0_range = (-np.pi, np.pi)
                self.physical_1_range = (-np.pi / 8, np.pi)

            elif domain == "point_mass":
                self.physical_0 = r"physical position (x)"
                self.physical_1 = r"physical position (y)"
                self.physical_0_range = (-0.3, 0.3)
                self.physical_1_range = (-0.3, 0.3)

            elif domain == "point_mass_3d":
                self.physical_0 = r"physical position (x)"
                self.physical_1 = r"physical position (y)"
                self.physical_2 = r"physical position (z)"
                self.physical_0_range = (-0.3, 0.3)
                self.physical_1_range = (-0.3, 0.3)
                self.physical_2_range = (-0.1, 0.4)

            else:
                assert False

        self.color_x = ["#22ff7a", "#e7ad38"]
        self.color_l = ["#16aa4f", "#c59330"]

    def set_axes_L0L1(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.latent_0, color=self.color_l[0])
            if lmax is not None:
                ax.set_xlim(-lmax, lmax)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

            ax.set_ylabel(self.latent_1, color=self.color_l[1])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)

    def set_axes_P0L0(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.physical_0, color=self.color_x[0])
            ax.set_xlim(*self.physical_0_range)

            ax.set_ylabel(self.latent_0, color=self.color_l[0])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)

    def set_axes_P1L1(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.physical_1, color=self.color_x[1])
            ax.set_xlim(*self.physical_1_range)

            ax.set_ylabel(self.latent_1, color=self.color_l[1])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)

    def set_axes_P2L2(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.physical_2, color=self.color_x[1])
            ax.set_xlim(*self.physical_2_range)

            ax.set_ylabel(self.latent_2, color=self.color_l[1])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)

    def set_axes_P0L1(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.physical_0, color=self.color_x[0])
            ax.set_xlim(*self.physical_0_range)

            ax.set_ylabel(self.latent_1, color=self.color_l[1])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)

    def set_axes_P1L0(self, ax: Axes, lmax: Optional[float] = None):
        if self.domain is not None:
            ax.set_xlabel(self.physical_1, color=self.color_x[1])
            ax.set_xlim(*self.physical_1_range)

            ax.set_ylabel(self.latent_0, color=self.color_l[0])
            if lmax is not None:
                ax.set_ylim(-lmax, lmax)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%3.1f"))

        mpu.Axis_aspect_2d(ax, 1)
