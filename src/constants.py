# Simulation parameters
NUM_AGENTS = 25
NUM_STEPS = 1000
NUM_INDICATORS = 12
NUM_PREDICTORS = 100
NUM_ASSETS = 1
MAX_AUCTION_ITERATIONS = 100
M = 1
C = 0.005
NUM_PREDICTORS_TOURNAMENT = 2
MODE = 0 # 0 = regular, 1 = predictors fixed to hree theoretical parameters

# hree theoretical parameters
HREE_A = 0.95
HREE_B = 4.501

SAVE_PREDICTORS = True # Store learned predictors in a file
LOAD_STATE = False # Load previously learned predictors from a file

INITIAL_DEFAULT_VARIANCE = 4.0

# Global Agent parameters

#FAST (Complex Regime)
REGIME = "FAST"
CROSSOVER_RATE = 0.1
GENETIC_EXPLORATION_PARAMETER = 250
LAMBDA = 1/75

# SLOW (Rational Regime)
# REGIME = "SLOW"
# CROSSOVER_RATE = 0.3
# GENETIC_EXPLORATION_PARAMETER = 1000
# LAMBDA = 1/150

AGENT_INITIAL_CASH = 20000
RISK_AVERSION     = 0.5

# Global Asset parameters
INTEREST_RATE = 0.1

# Asset 1 specific parameters
ASSET_1_SUPPLY = 25
ASSET_1_INITIAL_DIVIDEND = 10
ASSET_1_RHO = 0.95
ASSET_1_ERROR_VARIANCE = 0.0743