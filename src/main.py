from constants import *
from world import World
import numpy as np


SEED = 42
np.random.seed(SEED)


NUM_EXPERIMENTS = 1

def main():
    for experiment_id in range(1, NUM_EXPERIMENTS+1):
        print(f"Running experiment {experiment_id}...")
        parameter_filename = "data/slow/predictors.json"
        price_history_filename = "data/slow/SLOW_250000_price_history.csv"
        world = World(experiment_id=experiment_id, parameter_filename=parameter_filename, pricehistory_filename=price_history_filename)
        world.run()

if __name__ == "__main__":
    main()