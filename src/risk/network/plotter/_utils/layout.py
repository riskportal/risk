"""
risk/network/plotter/_utils/layout
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

from typing import Tuple

import numpy as np


def calculate_bounding_box(
    node_coordinates: np.ndarray, radius_margin: float = 1.05
) -> Tuple[Tuple, float]:
    """
    Calculate the bounding box of the network based on node coordinates.

    Args:
        node_coordinates (np.ndarray): Array of node coordinates (x, y).
        radius_margin (float, optional): Margin factor to apply to the bounding box radius. Defaults to 1.05.

    Returns:
        Tuple[Tuple, float]: Center (x, y) and radius of the bounding box.
    """
    # Find minimum and maximum x, y coordinates
    x_min, y_min = np.min(node_coordinates, axis=0)
    x_max, y_max = np.max(node_coordinates, axis=0)
    # Calculate the center of the bounding box
    center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    # Calculate the radius of the bounding box, adjusted by the margin
    radius = max(x_max - x_min, y_max - y_min) / 2 * radius_margin
    return center, radius
