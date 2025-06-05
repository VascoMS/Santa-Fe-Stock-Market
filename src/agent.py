import numpy as np
from constants import *
from predictor import Predictor
from typing import Any, Dict, List, Tuple

SEED = 42
np.random.seed(SEED)

class Agent:
    def __init__(self, id: str, cash: float):
        self._id = id
        # holdings in each of the three assets
        self._portfolio = {"asset_1": 0, "asset_2": 0, "asset_3": 0}
        # cash on hand
        self._cash = cash
        # desired holdings / demand each period
        self._demand = {"asset_1": 0, "asset_2": 0, "asset_3": 0}

        self._latest_observation = None

        self._activated_predictors: Dict[str, List[Predictor]] = {"asset_1": [], "asset_2": [], "asset_3": []}
        self._previous_activated_predictors: Dict[str, List[Predictor]] = {"asset_1": [], "asset_2": [], "asset_3": []}

        self._trading_volumes = []
        # create a pool of predictors per asset
        self._predictors: Dict[str, List[Predictor]] = {}
        for asset in self._portfolio:
            self._predictors[asset] = [
                Predictor(asset) for _ in range(NUM_PREDICTORS)
            ]
        self._default_predictor = Predictor.generate_default_predictor("asset_1")
        self._auction_beginning = True
        self._expected = 0
        self._latest_predictor = None

    def observe(self, observation: Dict[str, Any]) -> None:
        """
        Process and store an observation of the market state.
        
        Parameters:
        - observation: Dictionary containing:
            - bitstring: Binary array of market indicators
            - asset_indexes: Mapping of asset IDs to their indexes in the bitstring
            - prices: Current prices of all assets
            - dividends: Current dividends of all assets
            - uncleared_assets: List of assets with uncleared markets
        """
        self._latest_observation = observation

    def act(self) -> Dict[str, Tuple[int, float]]:
        """
        Calculate demands based on the latest observation.
        Returns a dictionary of asset_id -> (quantity, price_slope).
        """
        if not self._latest_observation:
            return {}
            
        bitstring = self._latest_observation["bitstring"]
        asset_indexes = self._latest_observation["asset_indexes"]
        prices = self._latest_observation["prices"]
        dividends = self._latest_observation["dividends"]
        uncleared_assets = self._latest_observation.get("uncleared_assets", list(asset_indexes.keys()))
        
        demands_and_slope = {}
        for asset in uncleared_assets:
            idx = asset_indexes[asset]
            price = prices[asset]
            dividend = dividends[asset]

            if self._auction_beginning:
                # Pick predictors that "fire" on the current bitstring
                active_predictors = [
                    p for p in self._predictors[asset]
                    if p.matches(bitstring[idx])
                ]
                
                if not active_predictors:
                    #print(f"Agent {self._id} found no active predictors for asset {asset}.")
                    active_predictors = [self._default_predictor]
                
                self._activated_predictors[asset] = active_predictors
                # Choose the most precise predictor
                best_p = min(active_predictors, key=lambda p: p.get_variance())
                # One-step ahead forecast of total payout
                self._expected = best_p.predict(price, dividend)
                self._latest_predictor = best_p
                #print(f"Agent {self._id} is computing new expected value and variance for asset {asset}: Expected = {self._expected}, Variance = {self._latest_predictor.get_variance()}")

            # CARA-optimal target shares
            target_h = (self._expected - price * (1 + INTEREST_RATE)) / (
                RISK_AVERSION * self._latest_predictor.get_variance()
            )
            qty = self._bound_demand(np.round(target_h, 2), self._portfolio[asset], price)

            # Record for portfolio update
            self._demand[asset] = qty
            # Submit to auction 
            slope = (self._latest_predictor.get_parameter_a() - (1 + INTEREST_RATE)) / (RISK_AVERSION * self._latest_predictor.get_variance()) # dh/dp
            #print(f"Agent {self._id} - Asset: {asset}, Expected: {self._expected}, Price: {price}, Demand: {qty}, Slope: {slope} a: {self._latest_predictor.get_parameter_a()}, b: {self._latest_predictor.get_parameter_b()} variance: {self._latest_predictor.get_variance()} bitstring: {self._latest_predictor._condition_string}")
            demands_and_slope[asset] = (qty, slope)
            
        return demands_and_slope
    
    def _bound_demand(self, demand, current_holding, price) -> None:
        delta = demand - current_holding
        if delta > 10:
            # If demand is too high, cap it to prevent excessive buying
            demand = current_holding + 10
        elif delta < -10:
            # If demand is too low, cap it to prevent excessive selling
            demand = current_holding - 10
        return min(max(demand, 0), self._cash/price + current_holding)
    
    def compute_wealth(self) -> float:
        """Calculate total wealth = cash + market value of all holdings."""
        if not self._latest_observation:
            return self._cash
            
        prices = self._latest_observation["prices"]
        
        portfolio_value = sum(
            self._portfolio[asset] * prices[asset]
            for asset in self._portfolio if asset in prices
        )
        
        return self._cash + portfolio_value

    def _update_cash(self) -> None:
        """Roll cash forward by interest and collect dividends."""
        if not self._latest_observation:
            return
            
        dividends = self._latest_observation["dividends"]
            
        # Dividends on holdings
        self._cash += sum(
            self._portfolio[asset] * dividends[asset]
            for asset in self._portfolio if asset in dividends
        )

    def _update_portfolio(self):
        """Re‐balance portfolio to match last period’s submitted demand."""
        asset_deltas = {asset: self._demand[asset] - self._portfolio[asset] for asset in self._portfolio}
        volume = sum(abs(delta) for delta in asset_deltas.values())
        self._trading_volumes.append(volume)
        self._cash -= sum(
            asset_deltas[asset] * self._latest_observation["prices"][asset]
            for asset in self._portfolio if asset in self._latest_observation["prices"]
        )
        self._portfolio = self._demand.copy()

    def _update_predictors(self) -> None:
        """
        Update each predictor's performance using the true price and dividend.
        Optionally: mutate or run GA here at rate 1/K.
        """
        if not self._latest_observation:
            return
            
        prices = self._latest_observation["prices"]
        dividends = self._latest_observation["dividends"]
        
        for asset in self._portfolio:
            if asset in prices and asset in dividends:
                true_price = prices[asset]
                true_dividend = dividends[asset]
                
                for predictor in self._previous_activated_predictors[asset]:
                    predictor.update(true_price, true_dividend)
                
                # Update the previous activated predictors
                self._previous_activated_predictors[asset] = self._activated_predictors[asset]
                self._activated_predictors[asset] = []
                    
                # Occasionally evolve predictors
                if np.random.rand() < (1.0 / GENETIC_EXPLORATION_PARAMETER) and MODE != 1:
                    self._evolve_predictors(asset)
    
    def update(self):
        """
        Update the agent's state.
        This method is called at each time step to update the agent's state.
        """
        # Update cash and portfolio
        self._update_cash()
        self._update_portfolio()
        
        # Update predictors
        self._update_predictors()

    def get_parent_for_evolution(self, asset: str, eligible_parents: List[Predictor]) -> Predictor:
        candidates = np.random.choice(eligible_parents, size=NUM_PREDICTORS_TOURNAMENT)
        return max(candidates, key=lambda p: p.calculate_fitness())
    
    def crossover(self, parent_1: Predictor, parent_2: Predictor) -> Predictor:
        crossover_type = np.random.choice([0, 1, 2])
        child = Predictor(parent_1.get_asset_name())
        if crossover_type == 0:
            # Uniform crossover
            parent_for_a = np.random.choice([parent_1, parent_2])
            child._a = parent_for_a.get_parameter_a()
            if parent_for_a == parent_1:
                child._b = parent_2.get_parameter_b()
            else:
                child._b = parent_1.get_parameter_b()
        elif crossover_type == 1:
            # Linear combination
            a_term = np.random.rand()
            b_term = np.random.rand()
            child._a = parent_1.get_parameter_a() * a_term + parent_2.get_parameter_a() * (1 - a_term)
            child._b = parent_1.get_parameter_b() * b_term + parent_2.get_parameter_b() * (1 - b_term)
        else:
            # Complete clone of one parent
            parent_choice = np.random.choice([parent_1, parent_2])
            child._a = parent_choice.get_parameter_a()
            child._b = parent_choice.get_parameter_b()
        
        for i in range(len(parent_1._condition_string)):
            if np.random.rand() < 0.5:
                child._condition_string[i] = parent_1._condition_string[i]
            else:
                child._condition_string[i] = parent_2._condition_string[i]
        child._variance = (parent_1.get_variance() + parent_2.get_variance()) / 2
        return child
            
            
    def _evolve_predictors(self, asset: str) -> None:
        """
        Run a genetic algorithm to evolve the predictors for a specific asset.
        Replaces the worst 20% (highest variance) with mutated clones of the top 20% (lowest variance).
        """
        # Sort predictors by variance (ascending: low to high)
        sorted_predictors = sorted(
            self._predictors[asset],
            key=lambda p: p.calculate_fitness()
        )

        num_to_replace = int(0.2 * len(sorted_predictors))

        # Select bottom 20% (lowest fitness)
        worst = sorted_predictors[:num_to_replace]

        eligible_parents = sorted_predictors[num_to_replace:]

        variances = [p.get_variance() for p in self._predictors[asset]]
        mean = np.mean(variances)

        for predictor in worst:
            #print(f"Agent {self._id} is evolving predictor : a: {predictor._a}, b: {predictor._b}, Condition: {predictor._condition_string}, Variance: {predictor.get_variance()}")

            if np.random.rand() <= CROSSOVER_RATE:
                new_predictor = self.crossover(
                    self.get_parent_for_evolution(asset, eligible_parents),
                    self.get_parent_for_evolution(asset, eligible_parents),
                )
            else:
                new_predictor = self.get_parent_for_evolution(asset, eligible_parents).clone()
                # new_predictor = Predictor(asset)
            if np.random.rand() < 0.03:
                new_predictor.mutate_params()
            bit_mutated = False
            for i in range(len(new_predictor._condition_string)):
                if np.random.rand() < 0.03:
                    new_predictor._condition_string[i] = np.random.choice(['0', '1', '#'])
                    bit_mutated = True
            if bit_mutated:
                new_predictor._variance = mean
            elif new_predictor._variance < self._default_predictor._variance - np.std(variances):
                new_predictor._variance = np.median(variances)

            self._predictors[asset].remove(predictor)
            self._predictors[asset].append(new_predictor)
        
