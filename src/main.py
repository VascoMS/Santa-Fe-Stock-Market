from constants import *
from world import World
import numpy as np


SEED = 42
np.random.seed(SEED)


def main():
    world = World()
    world.run()

if __name__ == "__main__":
    main()