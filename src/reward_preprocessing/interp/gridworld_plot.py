# Adapted from:
# https://github.com/HumanCompatibleAI/evaluating-rewards/blob/7b99ec9b415d805bd77041f2f7807d112dec9802/src/evaluating_rewards/analysis/reward_figures/gridworld_reward_heatmap.py
# Original copyright and license:

# Copyright 2020 Adam Gleave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import enum
import functools
import itertools
import math
from typing import Iterable, Mapping, Optional, Tuple
from unittest import mock

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


class Actions(enum.IntEnum):
    STAY = 0
    LEFT = 1
    UP = 2
    RIGHT = 3
    DOWN = 4


# (x,y) offset caused by taking an action
ACTION_DELTA = {
    Actions.STAY: (0, 0),
    Actions.LEFT: (-1, 0),
    Actions.UP: (0, 1),
    Actions.RIGHT: (1, 0),
    Actions.DOWN: (0, -1),
}

# Counter-clockwise, corners of a unit square, centred at (0.5, 0.5).
CORNERS = [(0, 0), (0, 1), (1, 1), (1, 0)]
# Vertices subdividing the unit square for each action
OFFSETS = {
    # Triangles, cutting unit-square into quarters
    direction: np.array(
        [CORNERS[action.value - 1], [0.5, 0.5], CORNERS[action.value % len(CORNERS)]]
    )
    for action, direction in ACTION_DELTA.items()
}
# Circle at the center
OFFSETS[(0, 0)] = np.array([0.5, 0.5])


def _set_ticks(n: int, subaxis: matplotlib.axis.Axis) -> None:
    subaxis.set_ticks(np.arange(0, n + 1), minor=True)
    subaxis.set_ticks(np.arange(n) + 0.5)
    subaxis.set_ticklabels(np.arange(n))


def _axis_formatting(ax: plt.Axes, xlen: int, ylen: int) -> None:
    """Construct figure and set sensible defaults."""
    # Axes limits
    ax.set_xlim(0, xlen)
    ax.set_ylim(0, ylen)
    # Make ticks centred in each cell
    _set_ticks(xlen, ax.xaxis)
    _set_ticks(ylen, ax.yaxis)
    # Draw grid along minor ticks, then remove those ticks so they don't protrude
    ax.grid(which="minor", color="k")
    ax.tick_params(which="minor", length=0, width=0)


def _reward_make_color_map(
    reward_arrays: Iterable[np.ndarray],
    vmin: Optional[float],
    vmax: Optional[float],
    normalizer=mcolors.Normalize,
) -> matplotlib.cm.ScalarMappable:
    if vmin is None:
        vmin = min(np.nanmin(arr) for arr in reward_arrays)
    if vmax is None:
        vmax = max(np.nanmax(arr) for arr in reward_arrays)
    norm = normalizer(vmin=vmin, vmax=vmax)
    return matplotlib.cm.ScalarMappable(norm=norm)


def _reward_draw_spline(
    x: int,
    y: int,
    action: int,
    optimal: bool,
    reward: float,
    from_dest: bool,
    mappable: matplotlib.cm.ScalarMappable,
    annot_padding: float,
    ax: plt.Axes,
) -> Tuple[np.ndarray, Tuple[float, ...], str]:
    # Compute shape position and color
    pos = np.array([x, y])
    direction = np.array(ACTION_DELTA[action])
    if from_dest:
        pos = pos + direction
        direction = -direction
    vert = pos + OFFSETS[tuple(direction)]
    color = mappable.to_rgba(reward)

    # Add annotation
    text = f"{reward:.0f}"
    lum = sns.utils.relative_luminance(color)
    text_color = ".15" if lum > 0.408 else "w"
    hatch_color = ".5" if lum > 0.408 else "w"
    xy = pos + 0.5

    if tuple(direction) != (0, 0):
        xy = xy + annot_padding * direction
    fontweight = "bold" if optimal else None
    # ax.annotate(
    #     text,
    #     xy=xy,
    #     ha="center",
    #     va="center_baseline",
    #     color=text_color,
    #     fontweight=fontweight,
    # )

    return vert, color, hatch_color


def _make_triangle(vert, color, **kwargs):
    return mpatches.Polygon(xy=vert, facecolor=color, **kwargs)


def _make_circle(vert, color, radius, **kwargs):
    return mpatches.Circle(xy=vert, radius=radius, facecolor=color, **kwargs)


def _reward_draw(
    state_action_reward: np.ndarray,
    discount: float,
    fig: plt.Figure,
    ax: plt.Axes,
    mappable: matplotlib.cm.ScalarMappable,
    from_dest: bool = False,
    edgecolor: str = "gray",
) -> None:
    """
    Draws a heatmap visualizing `state_action_reward` on `ax`.

    Args:
        state_action_reward: a three-dimensional array specifying the gridworld rewards.
        discount: MDP discount rate.
        fig: figure to plot on.
        ax: the axis on the figure to plot on.
        mappable: color map for heatmap.
        from_dest: if True, the triangular wedges represent reward when arriving into this
        cell from the adjacent cell; if False, represent reward when leaving this cell into
        the adjacent cell.
        edgecolor: color of edges.
    """
    # optimal_actions = optimal_mask(state_action_reward, discount)
    optimal_actions = np.zeros_like(state_action_reward, dtype=bool)

    circle_radius_pt = matplotlib.rcParams.get("font.size") * 0.7
    circle_radius_in = circle_radius_pt / 72
    corner_display = ax.transData.transform([0.0, 0.0])
    circle_radius_display = fig.dpi_scale_trans.transform([circle_radius_in, 0])
    circle_radius_data = ax.transData.inverted().transform(
        corner_display + circle_radius_display
    )
    annot_padding = 0.25 + 0.5 * circle_radius_data[0]

    triangle_patches = []
    circle_patches = []

    it = np.nditer(state_action_reward, flags=["multi_index"])
    while not it.finished:
        reward = it[0]
        x, y, action = it.multi_index
        optimal = optimal_actions[it.multi_index]
        it.iternext()

        if not np.isfinite(reward):
            assert action != 0
            continue

        vert, color, hatch_color = _reward_draw_spline(
            x, y, action, optimal, reward, from_dest, mappable, annot_padding, ax
        )

        hatch = "xx" if optimal else None
        if action == Actions.STAY:
            fn = functools.partial(_make_circle, radius=circle_radius_data[0])
        else:
            fn = _make_triangle
        patches = circle_patches if action == 0 else triangle_patches
        if hatch:  # draw the hatch using a different color
            patches.append(
                fn(vert, tuple(color), linewidth=1, edgecolor=hatch_color, hatch=hatch)
            )
            patches.append(
                fn(vert, tuple(color), linewidth=1, edgecolor=edgecolor, fill=False)
            )
        else:
            patches.append(fn(vert, tuple(color), linewidth=1, edgecolor=edgecolor))

    for p in triangle_patches + circle_patches:
        # need to draw circles on top of triangles
        ax.add_patch(p)


def plot_gridworld_rewards(
    reward_arrays: Mapping[str, np.ndarray],
    ncols: int,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cbar_format: str = "%.1f",
    cbar_fraction: float = 0.05,
    normalizer=mcolors.Normalize,
    **kwargs,
) -> plt.Figure:
    """
    Plots heatmaps of reward for the gridworld.

    Args:
      - reward_arrays: a mapping to three-dimensional arrays specifying the gridworld rewards.
      - ncols: number of columns per row.
      - vmin: the start of the color range; if unspecified, `min(reward_arrays)`.
      - vmax: the end of the color range; if unspecified, `max(reward_arrays)`.
      - cbar_format: format string for colorbar axis labels.
      - cbar_fraction: the size of the colorbar relative to a single heatmap.
      - **kwargs: passed through to `_reward_draw`.

    Returns:
        A Figure containing heatmaps for each array in `reward_arrays`. Each heatmap consists of
        a "supercell" for each state `(i,j)` in the original gridworld. This supercell contains a
        central circle, representing the no-op action reward and four triangular wedges,
        representing the left, up, right and down action rewards.
    """
    shapes = set((v.shape for v in reward_arrays.values()))
    assert len(shapes) == 1, "different shaped gridworlds cannot be in same plot"
    xlen, ylen, num_actions = next(iter(shapes))
    assert num_actions == len(ACTION_DELTA)

    nplots = len(reward_arrays)
    nrows = (nplots - 1) // ncols + 1
    width, height = matplotlib.rcParams.get("figure.figsize")
    fig = plt.figure(figsize=(width * ncols, height * nrows))
    width_ratios = [1] * ncols + [cbar_fraction]
    gs = fig.add_gridspec(nrows=nrows, ncols=ncols + 1, width_ratios=width_ratios)

    mappable = _reward_make_color_map(
        reward_arrays.values(), vmin, vmax, normalizer=normalizer
    )
    base_ax = fig.add_subplot(gs[0, 0])
    for idx, (pretty_name, reward) in enumerate(reward_arrays.items()):
        i = idx // ncols
        j = idx % ncols
        if i == 0 and j == 0:
            ax = base_ax
        else:
            ax = fig.add_subplot(gs[i, j], sharex=base_ax, sharey=base_ax)

        _axis_formatting(ax, xlen, ylen)
        if not ax.is_last_row():
            ax.tick_params(axis="x", labelbottom=False)
        if not ax.is_first_col():
            ax.tick_params(axis="y", labelleft=False)
        ax.set_title(pretty_name)

        _reward_draw(reward, fig=fig, ax=ax, mappable=mappable, **kwargs)

    for i in range(nrows):
        cax = fig.add_subplot(gs[i, -1])
        fig.colorbar(mappable=mappable, cax=cax, format=cbar_format)

    return fig


def prepare_rewards(rewards: torch.Tensor) -> np.ndarray:
    """Convert a tabular reward from the Mazelab format
    to the one needed for the plotting utilities.

    Args:
        rewards: (size ** 2, size ** 2) tensor, where size is the length
            of the gridworld. The first dimension indexes the state s,
            the second dimension the next state s'.

    Returns: (size, size, 5) array where the first two dimensions index
        the state and the last one the action.
    """
    # some sanity checks
    assert isinstance(rewards, torch.Tensor)
    assert rewards.ndim == 2
    assert rewards.size(0) == rewards.size(1)
    size = int(math.sqrt(rewards.size(0)))
    out = np.empty((size, size, 5))
    for i, j, a in itertools.product(range(size), range(size), range(5)):
        delta = ACTION_DELTA[Actions(a)]
        next_i = i + delta[0]
        next_j = j + delta[1]
        if not (0 <= next_i < size) or not (0 <= next_j < size):
            out[i, j, a] = np.nan
        else:
            state = i + j * size
            next_state = size * next_i + next_j
            out[i, j, a] = rewards[state, next_state].item()
    return out
