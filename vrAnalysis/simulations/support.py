import time
import numpy as np
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent


def get_box_coord(box_length, spacing=1):
    """
    Return the x and y coordinates of a box with a given length and spacing between points.

    args:
        box_length (int): the length of the box
        spacing (int): the spacing between points

    returns:
        xpos (np.ndarray): x coordinates of the box
        ypos (np.ndarray): y coordinates of the box
    """
    assert isinstance(spacing, int), "spacing must be an integer"
    assert isinstance(box_length, int), "box_length must be an integer"
    axis = np.arange(0, box_length, spacing)
    xpos, ypos = np.meshgrid(axis, axis)
    return xpos, ypos


def rand_centroid(box_length, grid=True):
    """
    Return a random centroid within a box of a given length.

    Will return a centroid on the grid if grid is set to True,
    otherwise, the centroid will be anywhere within the box.

    args:
        box_length (int): the length of the box
        grid (bool): whether the centroid should be on the grid or not

    returns:
        (tuple): the x and y coordinates of the centroid
                 returns integers if grid is True, otherwise floats
    """
    if grid:
        xc = np.random.randint(0, box_length)
        yc = np.random.randint(0, box_length)
    else:
        xc = np.random.random() * box_length
        xc = np.random.random() * box_length
    return xc, yc


def fwhm_to_var(x):
    """
    Return the variance of a Gaussian distribution given a desired FWHM.

    args:
        x (float): the desired FWHM

    returns:
        (float): the variance of the Gaussian distribution
    """
    return x**2 / (8 * np.log(2))


def get_place_map(xc, yc, xpos, ypos, place_width):
    """
    Create a place map given a centroid, place width, and room coordinates.

    args:
        xc (int): x coordinate of the centroid
        yc (int): y coordinate of the centroid
        xpos (np.ndarray): x coordinates of the room
        ypos (np.ndarray): y coordinates of the room
        place_width (float): the width of the place map

    returns:
        (np.ndarray): the place map
    """
    place_var = fwhm_to_var(place_width)
    numerator = -((xpos - xc) ** 2) - (ypos - yc) ** 2
    denominator = 2 * place_var
    return np.exp(numerator / denominator)


def get_grid_map(xc, yc, xpos, ypos, grid_spacing, grid_angle):
    """
    Generate a grid cell firing pattern using the Monaco & Abbott (2011) model.

    args:
        xc (int): x coordinate of the center
        yc (int): y coordinate of the center
        xpos (np.ndarray): x coordinates of the room
        ypos (np.ndarray): y coordinates of the room
        grid_spacing (float): the spacing between grid cells
        grid_angle (float): the angle of the grid

    returns:
        (np.ndarray): the grid cell firing pattern
    """
    theta = np.array([-np.pi / 3, 0, np.pi / 3])
    u = lambda theta: np.array([np.cos(theta), np.sin(theta)])
    spatial_offsets = np.stack((xpos - xc, ypos - yc), axis=2)
    scale_factor = (4 * np.pi) / (np.sqrt(3) * grid_spacing)
    cos_argument = np.array([np.cos(scale_factor * np.sum(spatial_offsets * u(theta[i] - grid_angle), axis=2)) for i in range(len(theta))])
    return np.sum(cos_argument, axis=0) / 3
