import time
import numpy as np
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from . import support
from .. import helpers


class Simulator:
    def __init__(
        self,
        box_length,
        spacing=1,
        dt=0.1,
        speed_mean=0.08,
        num_place_cells=50,
        num_grid_cells=50,
        place_width_mean=10,
        place_width_std=0.0,
        num_grid_modules=4,
        grid_expansion=1.3,
        base_grid_spacing=39.8,
        base_grid_width=27.4,
        g_noise_amp=1 / 100000,
    ):
        self.box_length = box_length
        self.spacing = spacing
        self.dt = dt
        self.speed_mean = speed_mean
        self.num_place_cells = num_place_cells
        self.num_grid_cells = num_grid_cells
        self.place_width_mean = place_width_mean
        self.place_width_std = place_width_std
        self.num_grid_modules = num_grid_modules
        self.grid_expansion = grid_expansion
        self.base_grid_spacing = base_grid_spacing
        self.base_grid_width = base_grid_width
        self.g_noise_amp = g_noise_amp

        # use parameters to create environment and agent
        self.set_environment()
        self.create_agent()
        self.create_cell_library()

    def set_environment(self):
        # get grid coordinates for a certain box size and spacing
        self.xgrid, self.ygrid = support.get_box_coord(self.box_length, spacing=self.spacing)
        # create a RatInABox environment
        self.env = Environment(dict(scale=self.box_length / 100))

    def create_agent(self):
        # Create agent with environment and parameters
        self.agent = Agent(self.env, dict(dt=self.dt, speed_mean=self.speed_mean))

    def create_cell_library(self):
        # Create place cell tuning curves
        if self.num_place_cells > 0:
            self.place_xc, self.place_yc = helpers.named_transpose([support.rand_centroid(self.box_length) for _ in range(self.num_place_cells)])
            self.place_width = np.random.normal(self.place_width_mean, self.place_width_std, self.num_place_cells)
            self.place_library = np.stack(
                [support.get_place_map(pxc, pyc, self.xgrid, self.ygrid, pw) for pxc, pyc, pw in zip(self.place_xc, self.place_yc, self.place_width)]
            )

        # Create grid cell tuning curves
        if self.num_grid_cells > 0:
            self.grid_xc, self.grid_yc = helpers.named_transpose([support.rand_centroid(self.box_length) for _ in range(self.num_grid_cells)])
            self.grid_spacing = np.array(
                [self.base_grid_spacing * (self.grid_expansion ** np.random.randint(0, self.num_grid_modules)) for _ in range(self.num_grid_cells)]
            )
            self.grid_angle = np.array([np.pi / 3 * np.random.random() for _ in range(self.num_grid_cells)])
            self.grid_library = np.stack(
                [
                    support.get_grid_map(gxc, gyc, self.xgrid, self.ygrid, gsp, gag)
                    for gxc, gyc, gsp, gag in zip(self.grid_xc, self.grid_yc, self.grid_spacing, self.grid_angle)
                ]
            )

    def run_simulation(self, n_steps=1000, reset=True):
        if reset:
            self.agent.reset_history()

        for _ in range(n_steps):
            self.agent.update()

        # Get trajectory variables
        t, pos = np.array(self.agent.history["t"]), np.array(self.agent.history["pos"])
        pos = pos * 100  # convert back to centimeters (to use as index)
        posidx = np.floor(pos).astype("int32")

        # Return
        return t, pos, posidx
