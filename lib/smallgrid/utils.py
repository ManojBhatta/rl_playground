from typing import Tuple
from .grid_world import GridWorld

def state_to_xy(s:int, env:GridWorld) -> Tuple[int]:
    x = s % env.env_size[0]
    y = s // env.env_size[0]
    return (x, y)

def xy_to_state(state:Tuple[int], env:GridWorld) -> int:
    x, y = state
    s  = y * env.env_size[0] + x
    return s
