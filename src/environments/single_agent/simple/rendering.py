import os
import sys

import numpy as np
import six
from gymnasium import error

from .simple import EnvState

if "Apple" in sys.version:
    if "DYLD_FALLBACK_LIBRARY_PATH" in os.environ:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] += ":/usr/lib"

import pyglet
from pyglet import shapes
from pyglet.gl import *


RAD2DEG = 57.29577951308232
# # Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)
_BLUE = (0, 0, 255)

_BACKGROUND_COLOR = _WHITE
_GRID_COLOR = _BLACK


def get_display(spec):
    """Convert a display specification (such as :0) into an actual Display
    object.
    Pyglet only supports multiple Displays on Linux.
    """
    if spec is None:
        return None
    elif isinstance(spec, six.string_types):
        return pyglet.canvas.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


class Viewer(object):
    def __init__(self, world_size):
        display = get_display(None)
        self.rows, self.cols = world_size

        self.grid_size = 100

        self.width = 1 + self.cols * (self.grid_size + 1)
        self.height = 1 + self.rows * (self.grid_size + 1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=display
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen = True

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen = False
        exit()

    def set_bounds(self, left, right, bottom, top):
        assert right > left and top > bottom
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        self.transform = Transform(
            translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        )

    def render(self, env_state: EnvState, return_rgb_array=False):
        glClearColor(*_WHITE, 0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid()
        self._draw_resources(env_state)
        self._draw_demonstrators(env_state)
        self._draw_ego_agent(env_state)

        arr = None
        if return_rgb_array:
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            arr = arr.reshape(buffer.height, buffer.width, 4)
            arr = arr[::-1, :, 0:3]

        self.window.flip()
        return arr if return_rgb_array else self.isopen

    def _draw_grid(self):
        batch = pyglet.graphics.Batch()
        v_lines, h_lines = [], []

        for r in range(self.rows + 1):
            v_lines.append(
                shapes.Line(
                    0,  # LEFT X
                    (self.grid_size + 1) * r + 1,  # Y
                    (self.grid_size + 1) * self.cols,  # RIGHT X
                    (self.grid_size + 1) * r + 1,  # Y,
                    thickness=2,
                    color=_BLACK,
                    batch=batch,
                )
            )

        for c in range(self.cols + 1):
            h_lines.append(
                shapes.Line(
                    (self.grid_size + 1) * c + 1,  # X
                    0,  # BOTTOM Y
                    (self.grid_size + 1) * c + 1,  # X
                    (self.grid_size + 1) * self.rows,  # TOP X
                    thickness=2,
                    color=_BLACK,
                    batch=batch,
                )
            )

        batch.draw()

    def _draw_resources(self, env_state: EnvState):
        # draw resources as green squares
        for r in range(self.rows + 1):
            for c in range(self.cols + 1):
                if env_state.resource_map[r][c] > 0:
                    resource_x = c * (self.grid_size + 1) + 1
                    resource_y = self.height - (self.grid_size + 1) * (r + 1) + 1
                    resource = shapes.Rectangle(
                        resource_x,
                        resource_y,
                        self.grid_size,
                        self.grid_size,
                        color=_GREEN,
                    )
                    resource.draw()

    def _draw_demonstrators(self, env_state: EnvState):
        # draw demonstrator agents as blue circles
        for agent_loc in env_state.demonstrator_locs:
            agent_x, agent_y = agent_loc
            agent_x = agent_x * (self.grid_size + 1) + 1
            agent_y = self.height - (self.grid_size + 1) * (agent_y + 1) + 1
            agent = shapes.Circle(
                agent_x + self.grid_size // 2,
                agent_y + self.grid_size // 2,
                self.grid_size // 2,
                color=_BLUE,
            )
            agent.draw()

    def _draw_ego_agent(self, env_state: EnvState):
        # draw ego agent as red circle
        agent_x, agent_y = env_state.ego_agent_loc
        agent_x = agent_x * (self.grid_size + 1) + 1
        agent_y = self.height - (self.grid_size + 1) * (agent_y + 1) + 1
        agent = shapes.Circle(
            agent_x + self.grid_size // 2,
            agent_y + self.grid_size // 2,
            self.grid_size // 2,
            color=_RED,
        )
        agent.draw()

    # def _draw_goal(self, env_state: EnvState):
    #     # draw goal as red square
    #     goal_x, goal_y = env_state.goal_loc
    #     goal_x = goal_x * (self.grid_size + 1) + 1
    #     goal_y = self.height - (self.grid_size + 1) * (goal_y + 1) + 1
    #     goal = shapes.Rectangle(
    #         goal_x,
    #         goal_y,
    #         self.grid_size,
    #         self.grid_size,
    #         color=_RED,
    #     )
    #     goal.draw()

    # def _draw_agent(self, env_state: EnvState):
    #     # draw agent as blue square
    #     agent_x, agent_y = env_state.agent_loc
    #     agent_x = agent_x * (self.grid_size + 1) + 1
    #     agent_y = self.height - (self.grid_size + 1) * (agent_y + 1) + 1
    #     agent = shapes.Rectangle(
    #         agent_x,
    #         agent_y,
    #         self.grid_size,
    #         self.grid_size,
    #         color=_BLUE,
    #     )
    #     agent.draw()
