"""
https://matplotlib.org/stable/tutorials/intermediate/arranging_axes.html

import mypython.plt_layout as pltl
"""

from typing import Callable, Generic, Iterator, List, Optional, TypeVar, Union

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from mpl_toolkits.mplot3d import Axes3D


class ContainerBase:
    def __init__(self, *, flex=1.0) -> None:
        self.children = None
        self._flex = flex


class Plotter(ContainerBase):
    def __init__(self, *, flex=1.0, clearable=True, **kwargs) -> None:
        """kwargs ex: , projection="3d" """
        super().__init__(flex=flex)

        self._clearable = clearable
        self._ax_kwargs = kwargs
        self._ax = None

    @property
    def ax(self) -> Optional[Union[Axes, Axes3D]]:
        return self._ax


class Space(ContainerBase):
    def __init__(self, *, flex=1.0) -> None:
        super().__init__(flex=flex)


_C = TypeVar("_C")


class _RowColumn(ContainerBase, Generic[_C]):
    def __init__(self, children: List[_C], *, flex=1.0, wspace=None, hspace=None) -> None:
        super().__init__(flex=flex)
        assert type(children) == list
        for child in children:
            if not isinstance(child, ContainerBase):
                raise TypeError(f"{child} ({type(child)})")

        self.children = children
        self._subfigs_kwargs = dict(wspace=wspace, hspace=hspace)


class Column(_RowColumn, Generic[_C]):
    """From right to left"""

    def __init__(self, children: List[_C], *, flex=1.0, space=None) -> None:
        super().__init__(children=children, flex=flex, hspace=space)
        self.children: List[_C]


class Row(_RowColumn, Generic[_C]):
    """From top to bottom"""

    def __init__(self, children: List[_C], *, flex=1.0, space=None) -> None:
        super().__init__(children=children, flex=flex, wspace=space)
        self.children: List[_C]


# TODO
# class Grid(Container):
#     def __init__(self) -> None:
#         super().__init__()


def compile(fig: Figure, container: ContainerBase):
    gs = fig.add_gridspec()

    if not isinstance(container, ContainerBase):
        raise TypeError(f"{container} ({type(container)})")

    _compile(fig, gs[0], container)

    return container


def _compile(fig: Figure, gs: SubplotSpec, container: ContainerBase):
    assert type(gs) == SubplotSpec

    if not isinstance(container, ContainerBase):
        raise TypeError(f"{container} ({type(container)})")

    type_containter = type(container)
    if type_containter == Plotter:
        container._ax = fig.add_subplot(gs, **container._ax_kwargs)

    elif type_containter == Row:
        if container.children is not None:
            ratios = [child._flex for child in container.children]
            gs_ = gs.subgridspec(
                1, len(container.children), width_ratios=ratios, **container._subfigs_kwargs
            )
            for i, child in enumerate(container.children):
                _compile(fig, gs_[i], child)

    elif type_containter == Column:
        if container.children is not None:
            ratios = [child._flex for child in container.children]
            gs_ = gs.subgridspec(
                len(container.children), 1, height_ratios=ratios, **container._subfigs_kwargs
            )
            for i, child in enumerate(container.children):
                _compile(fig, gs_[i], child)


def clear(containers: Union[ContainerBase, List[ContainerBase]]):
    type_containers = type(containers)
    if type_containers == list or type_containers == tuple:
        for container in containers:
            _clear(container)
    else:
        _clear(containers)


def _clear(container: ContainerBase):
    if not isinstance(container, ContainerBase):
        raise TypeError(f"{container} ({type(container)})")

    type_containter = type(container)

    if type_containter == Plotter:
        if container._clearable:
            container.ax.clear()
    elif container.children is not None:
        for child in container.children:
            clear(child)


if __name__ == "__main__":

    def ver1():
        fig = plt.figure()
        fig.suptitle("Title")

        ax1 = Plotter()
        # ax2 = Plotter(projection="3d")
        ax2 = Plotter()
        ax3 = Plotter()
        ax4 = Plotter()

        layout = Row(
            [
                ax1,
                ax2,
                Column(
                    [
                        ax3,
                        ax4,
                    ],
                    flex=0.5,
                ),
            ],
        )

        compile(fig, layout)

        # After compile

        # clear(layout)

        ax1.ax.set_title("ax1")
        # ...

        plt.show()

    def ver2():
        fig = plt.figure()
        fig.suptitle("Title")

        ax1 = Plotter()
        ax2 = Plotter()
        ax3 = Plotter()
        ax4 = Plotter()
        ax5 = Plotter()
        ax6 = Plotter()

        compile(fig, Column([ax1, ax2, Row([ax3, Column([ax4, ax5]), ax6])]))

        plt.show()

    ver1()
    # ver2()
