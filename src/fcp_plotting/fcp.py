"""
Four-Coordinate Paper (FCP) plotting utilities for pseudo-velocity shock analysis.

This module provides matplotlib helper functions to create properly formatted
Four-Coordinate Paper plots with logarithmic scales, oblique reference lines,
and appropriately positioned labels for shock response spectrum visualization.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.text import Text


G0_SI: float = 9.80665            # Standard gravity in m/s²
G0_IN: float = 386.08858267716535 # Standard gravity in in/s²


def _base_len_from_v_unit(v_unit: str) -> str:
    """
    Extract base length unit from velocity unit string.
    
    Parameters
    ----------
    v_unit : str
        Velocity unit string. Supported values:
        - 'm/s', 'ms', 'mps' for meters per second
        - 'in/s', 'ips' for inches per second
        
    Returns
    -------
    str
        Base length unit ('m' or 'in')
        
    Raises
    ------
    ValueError
        If v_unit is not a supported velocity unit
        
    Examples
    --------
    >>> _base_len_from_v_unit('m/s')
    'm'
    >>> _base_len_from_v_unit('in/s')
    'in'
    """
    if not isinstance(v_unit, str):
        raise ValueError(f"v_unit must be a string, got {type(v_unit)}")
        
    u = v_unit.lower()
    if u in {"m/s", "ms", "mps"}:
        return "m"
    if u in {"in/s", "ips"}:
        return "in"
    raise ValueError(f"v_unit must be 'm/s' or 'in/s', got '{v_unit}'")


def fcp(
    ax: Optional[Axes] = None,
    *,
    v_unit: str = "in/s",
) -> Axes:
    """
    Configure matplotlib axes for Four-Coordinate Paper (FCP) pseudo-velocity plots.

    Creates a logarithmic plot with frequency (Hz) on the x-axis and pseudo velocity
    on the y-axis. Adds oblique reference lines for constant acceleration (slope -1)
    and constant displacement (slope +1) with appropriately positioned and rotated
    decade labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        The matplotlib axes object to configure. If None, uses the current axes
        from `plt.gca()`.
    v_unit : str, default 'in/s'
        Velocity unit string that determines the displacement unit for labels.
        Supported values:
        - 'in/s', 'ips' : Uses inches for displacement labels
        - 'm/s', 'ms', 'mps' : Uses meters for displacement labels

    Returns
    -------
    matplotlib.axes.Axes
        The configured matplotlib axes object with logarithmic scaling and 
        reference lines added.

    Raises
    ------
    ValueError
        If `v_unit` is not a supported velocity unit string.
    TypeError
        If `v_unit` is not a string.

    Notes
    -----
    - The function sets both x and y axes to logarithmic scale
    - Reference lines are drawn across the full axis range
    - Acceleration labels appear on the right edge in units of 'g' 
    - Displacement labels appear on the left edge in base length units
    - Label rotations automatically adjust to match diagonal line angles
    - Multiple calls to this function will add additional reference lines
    
    .. important::
       This function must be called **after** setting axis limits with
       `ax.set_xlim()` and `ax.set_ylim()`. The function uses the current
       axis limits to determine reference line placement and labeling.
       Calling `fcp()` before setting limits may result in incorrect or
       missing reference lines.

    Examples
    --------
    Create a basic FCP plot with SI units:

    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from fcp_plotting import fcp
    >>> 
    >>> fig, ax = plt.subplots(figsize=(7, 5))
    >>> ax.set_xlim(1, 1000)
    >>> ax.set_ylim(1e-3, 10)
    >>> fcp(ax, v_unit='m/s')
    >>> 
    >>> # Add your PV data
    >>> f = np.logspace(0, 3, 100)
    >>> pv = 0.1 / (2*np.pi*f)  # Example: constant acceleration
    >>> ax.loglog(f, pv, 'b-', label='PV Data')
    >>> ax.legend()
    >>> plt.show()

    Create a plot with Imperial units:

    >>> fig, ax = plt.subplots(figsize=(7, 5))
    >>> ax.set_xlim(1, 1000)
    >>> ax.set_ylim(1e-2, 100)
    >>> fcp(ax, v_unit='in/s')
    >>> plt.show()
    """

    if ax is None:
        ax = plt.gca()

    base_len = _base_len_from_v_unit(v_unit)

    ax.set_xscale("log")
    ax.set_yscale("log")

    fmin, fmax = ax.get_xlim()
    vmin, vmax = ax.get_ylim()
    
    # Validate limits for logarithmic operations
    if fmin <= 0 or fmax <= 0 or vmin <= 0 or vmax <= 0:
        raise ValueError("All axis limits must be positive for logarithmic scales")
    if not all(math.isfinite(x) for x in [fmin, fmax, vmin, vmax]):
        raise ValueError("All axis limits must be finite")
        
    f_grid = np.logspace(np.log10(fmin), np.log10(fmax), 241)

    # Constants based on base length unit
    g0 = G0_SI if base_len == "m" else G0_IN

    # Analytic display angle for slope m on log–log axes
    def _display_angle_for_slope(m: float) -> float:
        """Calculate display angle in degrees for a line with given slope on log-log axes."""
        try:
            u0, u1 = math.log10(fmin), math.log10(fmax)
            v0, v1 = math.log10(vmin), math.log10(vmax)
            w = ax.bbox.width
            h = ax.bbox.height
            if w <= 0 or h <= 0:
                return 0.0
            scale = (h / w) * ((u1 - u0) / (v1 - v0))
            return math.degrees(math.atan(m * scale))
        except Exception:
            return 0.0

    def _decade_level_sets(min_val: float, max_val: float) -> Tuple[List[float], List[float]]:
        """
        Generate decade-based level sets for reference line placement.
        
        Returns
        -------
        Tuple[List[float], List[float]]
            (major_decades, all_levels) where major_decades contains powers of 10
            and all_levels contains all intermediate values (1-9 × 10^k).
        """
        eps = 1e-300
        lo = max(min_val, eps)
        hi = max(max_val, lo * 1.01)
        
        # Handle invalid values gracefully
        if not (math.isfinite(lo) and math.isfinite(hi) and lo > 0 and hi > 0):
            return [], []
            
        try:
            kmin = int(math.floor(math.log10(lo)))
            kmax = int(math.ceil(math.log10(hi)))
        except (ValueError, OverflowError):
            return [], []
            
        majors: List[float] = [10.0 ** k for k in range(kmin, kmax + 1)]
        all_levels: List[float] = []
        for k in range(kmin, kmax + 1):
            decade = 10.0 ** k
            for d in range(1, 10):
                all_levels.append(d * decade)
        return majors, all_levels

    # Store label angle update info so we can recompute after layout changes
    rot_items: List[Tuple[Text, float]] = []  # list of (Text, slope_m)

    # Helper: nudge label upward in display space (pixels)  
    def _nudge_up(x: float, y: float, pixels: float = 10.0) -> float:
        """Nudge a point upward by specified pixels in display space."""
        p = ax.transData.transform((x, y))
        p[1] += pixels
        y2 = ax.transData.inverted().transform(p)[1]
        return y2
    
    # Helper: check if label position is within bounds with buffer for text size
    def _is_label_within_bounds(x: float, y: float, xlim: Tuple[float, float], ylim: Tuple[float, float]) -> bool:
        """
        Check if a label position is within the axes bounds with buffer for text size.
        
        Args:
            x, y: Label position in data coordinates
            xlim, ylim: Axes limits (min, max)
        
        Returns:
            True if label (including text) will fit within bounds, False otherwise
        """
        # Add buffer to account for text size (especially for rotated text)
        # Use log-space calculations for log axes
        log_x_range = math.log10(xlim[1]) - math.log10(xlim[0])
        log_y_range = math.log10(ylim[1]) - math.log10(ylim[0])
        
        # Buffer: very small percentage of range in log space (converted back to linear)
        x_buffer_factor = 10**(0.01 * log_x_range)  # 1% buffer  
        y_buffer_factor = 10**(0.02 * log_y_range)  # 2% buffer for y due to text height
        
        x_min_buffered = xlim[0] * x_buffer_factor
        x_max_buffered = xlim[1] / x_buffer_factor
        y_min_buffered = ylim[0] * y_buffer_factor  
        y_max_buffered = ylim[1] / y_buffer_factor
        
        return (x_min_buffered <= x <= x_max_buffered and 
                y_min_buffered <= y <= y_max_buffered)
        
        # Clamp in log space with margins
        log_x_clamped = max(log_fmin + x_margin_factor, min(log_x, log_fmax - x_margin_factor))
        log_y_clamped = max(log_vmin + y_margin_factor, min(log_y, log_vmax - y_margin_factor))
        
        # Convert back to linear space
        x_clamped = 10 ** log_x_clamped
        y_clamped = 10 ** log_y_clamped
        
        return x_clamped, y_clamped

    # Determine dynamic level envelopes over the full frequency range.
    # For acceleration lines (a = 2πfv), we need the full range of accelerations
    # that intersect the plot area at any frequency within [fmin, fmax]
    a_candidates = [
        2 * math.pi * fmin * vmin,  # Bottom-left corner
        2 * math.pi * fmin * vmax,  # Top-left corner  
        2 * math.pi * fmax * vmin,  # Bottom-right corner
        2 * math.pi * fmax * vmax,  # Top-right corner
    ]
    a_min = min(a_candidates)
    a_max = max(a_candidates)
    
    # For displacement lines (d = v/(2πf)), we need the full range of displacements
    # that intersect the plot area at any frequency within [fmin, fmax]
    d_candidates = [
        vmin / (2 * math.pi * fmax),  # Bottom-left corner (smallest d)
        vmin / (2 * math.pi * fmin),  # Bottom-right corner
        vmax / (2 * math.pi * fmax),  # Top-left corner
        vmax / (2 * math.pi * fmin),  # Top-right corner (largest d)
    ]
    d_min = min(d_candidates)
    d_max = max(d_candidates)

    # Acceleration levels (majors for labels, all for lines)
    G_lo, G_hi = a_min / g0, a_max / g0
    acc_maj_G, acc_all_G = _decade_level_sets(G_lo, G_hi)
    acc_levels_all = [g * g0 for g in acc_all_G]
    acc_label_vals = [(g * g0, int(round(math.log10(g)))) for g in acc_maj_G]

    # Displacement levels (majors for labels, all for lines)
    disp_unit_name = "m" if base_len == "m" else "in"
    d_maj, d_all = _decade_level_sets(d_min, d_max)
    disp_levels_all = d_all
    disp_label_vals = [(d, int(round(math.log10(d)))) for d in d_maj]

    # Match reference lines to grid aesthetics by default
    lk = {"color": "0.8", "lw": 0.6, "ls": ":", "zorder": 0}
    tk = {"color": "0.35", "fontsize": 8, "ha": "center", "va": "center"}

    # Plot all acceleration lines across full x-range
    for a in acc_levels_all:
        fx = f_grid
        vy = a / (2 * math.pi * fx)
        ax.plot(fx, vy, **lk)

    # Label acceleration decades at right edge
    for a, n_exp in acc_label_vals:
        # Position close to right edge
        f_right = 10 ** (math.log10(fmax) - 0.02 * (math.log10(fmax) - math.log10(fmin)))
        v_right = a / (2 * math.pi * f_right)
        if not (vmin <= v_right <= vmax):
            continue
        
        # Nudge up and check if within bounds
        v_nudged = _nudge_up(f_right, v_right)
        
        # Skip this label if nudged position would be outside bounds
        if not _is_label_within_bounds(f_right, v_nudged, (fmin, fmax), (vmin, vmax)):
            continue
            
        ang = _display_angle_for_slope(-1.0)
        tkwargs = {k: v for k, v in tk.items() if k not in {"ha", "va"}}
        txt = ax.text(f_right, v_nudged, rf"$10^{{{n_exp}}}$ g", rotation=ang, ha="right", va="center", **tkwargs)
        try:
            txt.set_transform_rotates_text(False)
            txt.set_rotation_mode('anchor')
        except Exception:
            pass
        rot_items.append((txt, -1.0))

    # Plot all displacement lines across full x-range
    for d in disp_levels_all:
        fx = f_grid
        vy = 2 * math.pi * fx * d
        ax.plot(fx, vy, **lk)

    # Label displacement decades at left edge
    for d, n_exp in disp_label_vals:
        # Position close to left edge  
        f_left = 10 ** (math.log10(fmin) + 0.02 * (math.log10(fmax) - math.log10(fmin)))
        v_left = 2 * math.pi * f_left * d
        if not (vmin <= v_left <= vmax):
            continue
        
        # Nudge up and check if within bounds
        v_nudged = _nudge_up(f_left, v_left)
        
        # Skip this label if nudged position would be outside bounds
        if not _is_label_within_bounds(f_left, v_nudged, (fmin, fmax), (vmin, vmax)):
            continue
            
        ang = _display_angle_for_slope(+1.0)
        tkwargs = {k: v for k, v in tk.items() if k not in {"ha", "va"}}
        txt = ax.text(f_left, v_nudged, rf"$10^{{{n_exp}}}$ {disp_unit_name}", rotation=ang, ha="left", va="center", **tkwargs)
        try:
            txt.set_transform_rotates_text(False)
            txt.set_rotation_mode('anchor')
        except Exception:
            pass
        rot_items.append((txt, +1.0))

    # Update rotations after draw to account for layout/resizes
    def _update_label_rotations(event=None):
        for t, m in rot_items:
            ang = _display_angle_for_slope(m)
            t.set_rotation(ang)
            try:
                t.set_transform_rotates_text(False)
                t.set_rotation_mode('anchor')
            except Exception:
                pass

    fig = ax.figure
    if fig is not None and hasattr(fig, 'canvas') and fig.canvas is not None:
        cid_attr = '_fcp_rot_cid'
        if getattr(ax, cid_attr, None) is not None:
            try:
                fig.canvas.mpl_disconnect(getattr(ax, cid_attr))
            except Exception:
                pass
        cid = fig.canvas.mpl_connect('draw_event', _update_label_rotations)
        setattr(ax, cid_attr, cid)
        try:
            _update_label_rotations()
        except Exception:
            pass

    ax.set_zorder(1)
    ax.patch.set_visible(False)
    return ax

