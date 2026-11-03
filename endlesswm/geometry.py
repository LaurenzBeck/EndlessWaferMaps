"""# 🟩⭕ Geometric Shapes

This module contains functions that create and add geometric shapes to arrays.
"""

import math

import numpy as np
from skimage import draw


def rectangle(start: tuple[int, int], extent: tuple[int, int], shape: tuple[int, int]) -> np.array:
    """🟧 creates an array with a rectangle.

    Args:
        start (tuple[int, int]): coordinates of the top-left corner
        extent (tuple[int, int]): height and width
        shape (tuple[int, int]): wafer map shape

    Returns:
        np.array: array of dimensions `shape`
            with `1` inside and `0` outside the rectangle
    """
    array = np.zeros(shape, dtype=np.uint8)
    rr, cc = draw.rectangle(start, extent=extent, shape=shape)
    array[rr, cc] = 1
    return array


def circle(center: tuple[int, int], diameter: int, shape: tuple[int, int]) -> np.array:
    """🔴 creates an array with a circle.

    Args:
        center (tuple[int, int]): center coordinates of the circle
        diameter (int): diameter of the circle
        shape (tuple[int, int]): wafer map shape

    Returns:
        np.array: array of dimensions `shape`
            with `1` inside and `0` outside the circle
    """
    array = np.zeros(shape, dtype=np.uint8)
    # Compute semi-axes for ellipse
    axis_y = diameter / 2  # Semi-axis for height
    axis_x = diameter / 2 * (shape[1] / shape[0])  # Semi-axis adjusted for width aspect ratio
    rr, cc = draw.ellipse(center[0], center[1], axis_y, axis_x, shape=shape)
    array[rr, cc] = 1
    return array


def ring(center: tuple[int, int], diameter: int, width: int, shape: tuple[int, int]) -> np.array:
    """💍 creates an array with a ring.

    Args:
        center (tuple[int, int]): center coordinates of the ring
        diameter (int): diameter of the ring
        width (int): width of the ring
        shape (tuple[int, int]): wafer map shape

    Returns:
        np.array: array of dimensions `shape`
            with `1` inside and `0` outside the ring
    """
    array = np.zeros(shape, dtype=np.uint8)
    # Compute semi-axes for the outer and inner ellipses
    outer_axis_y = diameter / 2  # Outer semi-axis for height
    outer_axis_x = diameter / 2 * (shape[1] / shape[0])  # Outer semi-axis adjusted for width
    inner_axis_y = max(0, outer_axis_y - width)  # Inner semi-axis for height
    inner_axis_x = max(0, outer_axis_x - round(width * (shape[1] / shape[0])))  # Inner semi-axis adjusted for width
    # Draw the outer ellipse
    rr, cc = draw.ellipse(center[0], center[1], outer_axis_y, outer_axis_x, shape=shape)
    array[rr, cc] = 1
    # Subtract the inner ellipse
    rr, cc = draw.ellipse(center[0], center[1], inner_axis_y, inner_axis_x, shape=shape)
    array[rr, cc] = 0
    return array


def crescent(angle: float, max_width: float, shape: tuple[int, int], **kwargs) -> np.array:
    """🌒 creates an array with an elliptical crescent pattern.

    Args:
        angle (float): angle (in radians) defining the direction of the crescent.
        max_width (float): crescent width as a fraction of the min(shape) at the widest part.
        shape (tuple[int, int]): shape of the array (height, width).

    Returns:
        np.array: array of dimensions `shape` with `1` inside and `0` outside the crescent.
    """
    max_width = max_width * min(*shape)

    array = full_ellipse(shape)
    center_y, center_x = shape[0] / 2, shape[1] / 2

    shift_x = np.cos(angle) * max_width
    shift_y = np.sin(angle) * max_width
    rr, cc = draw.ellipse(center_y + shift_y, center_x + shift_x, center_y, center_x, shape=shape)
    array[rr, cc] = 0

    return array


def full_circle(diameter: int) -> np.array:
    """🔴 creates an array with one big enclosed circle.

    Args:
        diameter (int): diameter of the circle

    Returns:
        np.array: array of dimensions (diameter, diameter)
            with `1` inside and `0` outside the circle
    """
    array = np.zeros((diameter, diameter), dtype=np.uint8)
    rr, cc = draw.disk((diameter // 2, diameter // 2), diameter // 2 + 1, shape=(diameter, diameter))
    array[rr, cc] = 1
    return array


def full_ellipse(shape: tuple[int, int]) -> np.array:
    """🔴 creates an array with one big enclosed ellipse.

    Args:
        shape (tuple[int, int]): enclosing shape of the ellipse

    Returns:
        np.array: array of dimensions `shape`
            with `1` inside and `0` outside the ellipse
    """
    array = np.zeros(shape, dtype=np.uint8)
    rr, cc = draw.ellipse(shape[0] / 2, shape[1] / 2, shape[0] / 2 + 1, shape[1] / 2 + 1, shape=shape)
    array[rr, cc] = 1
    return array


def line(start: tuple[int, int], end: tuple[int, int], shape: tuple[int, int]) -> np.array:
    """📏 creates an array with a line from start to end.

    Args:
        start (tuple[int, int]): start point
        end (tuple[int, int]): end point
        shape (tuple[int, int]): wafer map shape

    Returns:
        np.array: array of dimensions `shape`
            with `1` inside and `0` outside the line
    """
    array = np.zeros(shape, dtype=np.uint8)
    rr, cc = draw.line(*start, *end)
    array[rr, cc] = 1
    return array


def curved_line(
    start: tuple[int, int],
    middle: tuple[int, int],
    end: tuple[int, int],
    shape: tuple[int, int],
    weight: float = 1.0,
) -> np.array:
    """📏 creates an array with a curved bezier line from start to end through the mid point.

    Args:
        start (tuple[int, int]): start point
        middle (tuple[int, int]): middle point
        end (tuple[int, int]): end point
        shape (tuple[int, int]): wafer map shape
        weight (float, optional): middle control point weight, it describes the line tension.
            Defaults to 1.

    Returns:
        np.array: array of dimensions `shape`
            with `1` inside and `0` outside the line
    """
    array = np.zeros(shape, dtype=np.uint8)
    rr, cc = draw.bezier_curve(*start, *middle, *end, weight)
    array[rr, cc] = 1
    return array


def broad_line(start: tuple[int, int], end: tuple[int, int], shape: tuple[int, int], width: int = 1) -> np.array:
    """📏 creates an array with a broad line from start to end with a certain width.

    Args:
        start (tuple[int, int]): start point
        end (tuple[int, int]): end point
        shape (tuple[int, int]): wafer map shape
        width (int, optional): line width in pixel

    Returns:
        np.array: array of dimensions `shape`
            with `1` inside and `0` outside the line
    """
    array = line(start, end, shape)
    points = [(start, end), (start, end)]
    width -= 1

    while width > 0:
        if width % 2 == 0:
            # add line above
            # get last point from above
            (x1, y1), (x2, y2) = points[-2]

            # calculate the direction vector
            dx = x2 - x1
            dy = y2 - y1

            # calculate the length of the direction vector
            length = math.sqrt(dx**2 + dy**2)

            # Normalize the direction vector
            unit_dx = dx / length
            unit_dy = dy / length

            # Calculate the perpendicular vector (rotated 90 degrees clockwise)
            perp_unit_dx = unit_dy
            perp_unit_dy = -unit_dx

            scaled_perp_dx = round(perp_unit_dx)
            scaled_perp_dy = round(perp_unit_dy)
            if abs(scaled_perp_dx) == 1 and abs(scaled_perp_dy) == 1:
                scaled_perp_dy = 0

            # Calculate the new points
            new_start = (
                np.clip(x1 + scaled_perp_dx, a_min=0, a_max=shape[0] - 1),
                np.clip(y1 + scaled_perp_dy, a_min=0, a_max=shape[1] - 1),
            )
            new_end = (np.clip(x2 + scaled_perp_dx, a_min=0, a_max=shape[0] - 1), np.clip(y2 + scaled_perp_dy, a_min=0, a_max=shape[1] - 1))

            array += line(new_start, new_end, shape)

            # shift point by one pixel
            points.append((new_start, new_end))
            width -= 1
        else:
            # add line below
            # get last point from below
            (x1, y1), (x2, y2) = points[-2]

            # calculate the direction vector
            dx = x2 - x1
            dy = y2 - y1

            # calculate the length of the direction vector
            length = math.sqrt(dx**2 + dy**2)

            # Normalize the direction vector
            unit_dx = dx / length
            unit_dy = dy / length

            # Calculate the perpendicular vector (rotated 90 degrees counter-clockwise)
            perp_unit_dx = -unit_dy  # * the only difference to above is the minus sign here
            perp_unit_dy = unit_dx

            scaled_perp_dx = round(perp_unit_dx)
            scaled_perp_dy = round(perp_unit_dy)
            if abs(scaled_perp_dx) == 1 and abs(scaled_perp_dy) == 1:
                scaled_perp_dy = 0

            # Calculate the new points
            new_start = (
                np.clip(x1 + scaled_perp_dx, a_min=0, a_max=shape[0] - 1),
                np.clip(y1 + scaled_perp_dy, a_min=0, a_max=shape[1] - 1),
            )
            new_end = (np.clip(x2 + scaled_perp_dx, a_min=0, a_max=shape[0] - 1), np.clip(y2 + scaled_perp_dy, a_min=0, a_max=shape[1] - 1))

            array += line(new_start, new_end, shape)

            # shift point by one pixel
            points.append((new_start, new_end))
            width -= 1

    array[array > 1] = 1

    return array


def thick_line(start: tuple[int, int], end: tuple[int, int], shape: tuple[int, int], width: int = 1) -> np.array:
    """📏 creates an array with a thick line from start to end with a certain width.

    Args:
        start (tuple[int, int]): start point
        end (tuple[int, int]): end point
        shape (tuple[int, int]): wafer map shape
        width (int, optional): line width in pixel

    Returns:
        np.array: array of dimensions `shape`
            with `1` inside and `0` outside the line
    """
    array = np.zeros(shape, dtype=np.uint8)
    rr, cc = draw.line(*start, *end)
    radius = max(0.5, float(width) / 2.0)

    for r, c in zip(rr, cc, strict=False):
        circle_rr, circle_cc = draw.disk((r, c), radius, shape=shape)
        array[circle_rr, circle_cc] = 1

    return array


def broad_curved_line(
    start: tuple[int, int],
    middle: tuple[int, int],
    end: tuple[int, int],
    shape: tuple[int, int],
    width: int = 1,
) -> np.array:
    """📏 creates an array with a curved bezier line from start to end through the mid point with a certain width.

    Args:
        start (tuple[int, int]): start point
        middle (tuple[int, int]): middle point
        end (tuple[int, int]): end point
        shape (tuple[int, int]): wafer map shape
        width (int, optional): line width in pixel

    Returns:
        np.array: array of dimensions `shape`
            with `1` inside and `0` outside the line
    """
    array = curved_line(start, middle, end, shape)
    points = [(start, middle, end), (start, middle, end)]
    width -= 1

    while width > 0:
        if width % 2 == 0:
            # add line above
            # get last point from above
            (x1, y1), (xm, ym), (x2, y2) = points[-2]

            # calculate the direction vector
            dx = x2 - x1
            dy = y2 - y1

            # calculate the length of the direction vector
            length = math.sqrt(dx**2 + dy**2)

            # Normalize the direction vector
            unit_dx = dx / length
            unit_dy = dy / length

            # Calculate the perpendicular vector (rotated 90 degrees clockwise)
            perp_unit_dx = unit_dy
            perp_unit_dy = -unit_dx

            scaled_perp_dx = round(perp_unit_dx)
            scaled_perp_dy = round(perp_unit_dy)
            if abs(scaled_perp_dx) == 1 and abs(scaled_perp_dy) == 1:
                scaled_perp_dy = 0

            # Calculate the new points
            new_start = (
                np.clip(x1 + scaled_perp_dx, a_min=0, a_max=shape[0] - 1),
                np.clip(y1 + scaled_perp_dy, a_min=0, a_max=shape[1] - 1),
            )
            new_middle = (
                np.clip(xm + scaled_perp_dx, a_min=0, a_max=shape[0] - 1),
                np.clip(ym + scaled_perp_dy, a_min=0, a_max=shape[1] - 1),
            )
            new_end = (np.clip(x2 + scaled_perp_dx, a_min=0, a_max=shape[0] - 1), np.clip(y2 + scaled_perp_dy, a_min=0, a_max=shape[1] - 1))

            array += curved_line(new_start, new_middle, new_end, shape)

            # shift point by one pixel
            points.append((new_start, new_middle, new_end))
            width -= 1
        else:
            # add line below
            # get last point from below
            (x1, y1), (xm, ym), (x2, y2) = points[-2]

            # calculate the direction vector
            dx = x2 - x1
            dy = y2 - y1

            # calculate the length of the direction vector
            length = math.sqrt(dx**2 + dy**2)

            # Normalize the direction vector
            unit_dx = dx / length
            unit_dy = dy / length

            # Calculate the perpendicular vector (rotated 90 degrees counter-clockwise)
            perp_unit_dx = -unit_dy  # * the only difference to above is the minus sign here
            perp_unit_dy = unit_dx

            scaled_perp_dx = round(perp_unit_dx)
            scaled_perp_dy = round(perp_unit_dy)
            if abs(scaled_perp_dx) == 1 and abs(scaled_perp_dy) == 1:
                scaled_perp_dy = 0

            # Calculate the new points
            new_start = (
                np.clip(x1 + scaled_perp_dx, a_min=0, a_max=shape[0] - 1),
                np.clip(y1 + scaled_perp_dy, a_min=0, a_max=shape[1] - 1),
            )
            new_middle = (
                np.clip(xm + scaled_perp_dx, a_min=0, a_max=shape[0] - 1),
                np.clip(ym + scaled_perp_dy, a_min=0, a_max=shape[1] - 1),
            )
            new_end = (np.clip(x2 + scaled_perp_dx, a_min=0, a_max=shape[0] - 1), np.clip(y2 + scaled_perp_dy, a_min=0, a_max=shape[1] - 1))

            array += curved_line(new_start, new_middle, new_end, shape)

            # shift point by one pixel
            points.append((new_start, new_middle, new_end))
            width -= 1

    array[array > 1] = 1

    return array


def thick_curved_line(
    start: tuple[int, int],
    middle: tuple[int, int],
    end: tuple[int, int],
    shape: tuple[int, int],
    width: int = 1,
) -> np.array:
    """📏 creates an array with a curved bezier line from start to end through the mid point with a certain width.

    Args:
        start (tuple[int, int]): start point
        middle (tuple[int, int]): middle point
        end (tuple[int, int]): end point
        shape (tuple[int, int]): wafer map shape
        width (int, optional): line width in pixel

    Returns:
        np.array: array of dimensions `shape`
            with `1` inside and `0` outside the line
    """
    array = np.zeros(shape, dtype=np.uint8)
    rr, cc = draw.bezier_curve(*start, *middle, *end, weight=1.0)
    radius = max(0.5, float(width) / 2.0)

    for r, c in zip(rr, cc, strict=False):
        circle_rr, circle_cc = draw.disk((r, c), radius, shape=shape)
        array[circle_rr, circle_cc] = 1

    return array


def linear_gradient(shape: tuple[int, int], angle: float, slant_fraction: float) -> np.array:
    """📏 creates an array with a symmetric linear gradient around `1.0`.

    Args:
        shape (tuple[int, int]): shape of the gradient
        angle (float): angle (in rad) of the direction vector (dx, dy)
        slant_fraction (float): half-width of the gradient range around `1.0`

    Returns:
        np.array: array with `shape` and a linear gradient from
            `1.0 - slant_fraction` to `1.0 + slant_fraction`
    """
    min_val = 1.0 - slant_fraction
    max_val = 1.0 + slant_fraction

    # normalize the direction vector
    direction = (np.cos(angle), np.sin(angle))
    direction = np.array(direction) / np.linalg.norm(direction)

    # create a 2D grid of coordinates for x and y
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))

    # calculate the gradient values
    gradient = direction[0] * x + direction[1] * y

    # scale, shift and transpose the gradient values
    return ((gradient - gradient.min()) / (gradient.max() - gradient.min()) * (max_val - min_val) + min_val).T


def linear_radial_gradient(shape: tuple[int, int], center_val: float, outer_val: float) -> np.array:
    """⭕ creates an array with a linear radial gradient from `center_val` to `outer_val`.

    Args:
        shape (tuple[int, int]): shape of the gradient
        center_val (float): value at the center of the gradient
        outer_val (float): value at the outer edge of the gradient

    Returns:
        np.array: array with `shape` and a linear radial gradient from `center_val` to `outer_val`
    """
    center_x = shape[0] / 2
    center_y = shape[1] / 2

    # Compute the distances to the middle points of the edges
    max_dist_top_bottom = center_x  # Distance to top/bottom edge midpoints
    max_dist_left_right = center_y  # Distance to left/right edge midpoints

    # Initialize a grid of coordinates
    y_indices, x_indices = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    # Compute the radial distance for each point from the center
    dist_x = abs(x_indices - center_x) / max_dist_top_bottom  # Normalized horizontal distance
    dist_y = abs(y_indices - center_y) / max_dist_left_right  # Normalized vertical distance

    # The radial distance is determined by the Euclidean distance (scaled properly to account for rectangle)
    radial_dist = np.sqrt(dist_x**2 + dist_y**2)

    # Normalize the radial distance so that the maximum value is 1.0
    radial_dist = np.clip(radial_dist, 0, 1)

    # Scale the gradient to the desired range [center_val, outer_val]
    if center_val < outer_val:
        return radial_dist * (outer_val - center_val) + center_val
    return (1 - radial_dist) * (center_val - outer_val) + outer_val
