# agent.py

import numpy as np
from market_maker import MarketMaker
from constants import *
from predictor import Predictor
from typing import Dict, List, Tuple

class Agent:
    def __init__(self, id: str, cash: float, ):
        self._id = id
        # holdings in each of the three assets
        self._portfolio = {"asset_1": 0, "asset_2": 0, "asset_3": 0}
        # cash on hand
        self._cash = cash
        # desired holdings / demand each period
        self._demand = {"asset_1": 0, "asset_2": 0, "asset_3": 0}

        # create a pool of predictors per asset
        self._predictors: Dict[str, List[Predictor]] = {}
        for asset in self._portfolio:
            self._predictors[asset] = [
                Predictor(asset) for _ in range(NUM_PREDICTORS)
            ]

    def compute_wealth(self) -> float:
        """Total wealth = cash + market value of all holdings."""
        return self._cash + sum(
            self._portfolio[a] * self._market_maker.get_price(a)
            for a in self._portfolio
        )

    def update_cash(self):
        """Roll cash forward by interest and collect dividends."""
        # interest on cash
        self._cash *= (1 + INTEREST_RATE)
        # dividends on holdings
        self._cash += sum(
            self._portfolio[a] * self._market_maker.get_dividend(a)
            for a in self._portfolio
        )

    def update_portfolio(self):
        """Re‐balance portfolio to match last period’s submitted demand."""
        self._portfolio = self._demand.copy()

    def calc_demands(self, bitstring: np.ndarray, asset_indexes: dict, prices: Dict[str, float], dividends: Dict[str, float]) -> Dict[str, Tuple[int, float]]:
        """
        Given the current world‐state bits, for each asset:
          1. Filter predictors whose condition string matches.
          2. Pick the lowest‐variance predictor.
          3. Compute CARA‐optimal holding: h* = (E[price+div] - p(1+r)) / (λ·Var)
          4. Round to integer shares, store in self._demand,
             and send (demand, slope=dh*/dp) to the market maker.
        """
        demands_and_slope = dict()
        for asset, idx in asset_indexes.items():
            price = prices[asset]
            dividend = dividends[asset]

            # pick predictors that “fire” on the current bitstring
            active_predictors = [
                p for p in self._predictors[asset]
                if p.matches(bitstring[idx])
            ]
            if not active_predictors:
                # no signal ⇒ no position
                self._demand[asset] = 0
                demands_and_slope[asset] = 0, 0
                continue

            # choose the most precise predictor
            best_p = min(active_predictors, key=lambda p: p.get_variance())

            # one‐step ahead forecast of total payout
            expected = best_p.predict(price, dividend)
            variance = best_p.get_variance()

            # CARA‐optimal target shares
            target_h = (expected - price * (1 + INTEREST_RATE)) / (
                RISK_AVERSION * variance
            )
            qty = int(np.round(target_h))

            # record for portfolio update
            self._demand[asset] = qty
            # submit to auction 
            slope = (best_p.get_parameter_a() - (1 + INTEREST_RATE)) / (LAMBDA * variance) # Derivative of the demand function with respect to price
            demands_and_slope[asset] = qty, slope
        return demands_and_slope
    

    def update_predictors(self):
        """
        Update each predictor’s performance using the true price and dividend.
        Optionally: mutate or run GA here at rate 1/K.
        """
        for asset in self._portfolio:
            true_price = self._market_maker.get_price(asset)
            true_dividend = self._market_maker.get_dividend(asset)
            for predictor in self._predictors[asset]:
                predictor.update(true_price, true_dividend)
            if np.random.rand() < 1 / GENETIC_EXPLORATION_PARAMETER:
                self.genetic_algorithm()
            
    def genetic_algorithm(self):
        """
        Run a genetic algorithm to evolve the predictors.
        Replaces the worst 20% (highest variance) with mutated clones of the top 20% (lowest variance).
        """
        # Sort predictors by variance (ascending: low to high)
        sorted_predictors = sorted(
            self._predictors.values(),
            key=lambda p: p.get_variance()
        )

        num_to_replace = int(0.2 * len(sorted_predictors))

        # Select top 20% (lowest variance)
        best = sorted_predictors[:num_to_replace]

        # Select bottom 20% (highest variance)
        worst = sorted_predictors[-num_to_replace:]

        # Create mutated clones of the best predictors
        mutated_clones = []
        for predictor in best:
            clone = predictor.clone()
            clone.mutate()
            mutated_clones.append(clone)

        # Remove the worst predictors
        remaining_predictors = [
            p for p in sorted_predictors if p not in worst
        ]

        # Combine remaining + mutated clones
        self._predictors = remaining_predictors + mutated_clones
