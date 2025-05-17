# agent.py

import numpy as np
from market_maker import MarketMaker
from constants import *
from predictor import Predictor

class Agent:
    def __init__(self, id: str, cash: float, market_maker: MarketMaker):
        self._id = id
        # holdings in each of the three assets
        self._portfolio = {"asset_1": 0, "asset_2": 0, "asset_3": 0}
        # cash on hand
        self._cash = cash
        # desired holdings / demand each period
        self._demand = {"asset_1": 0, "asset_2": 0, "asset_3": 0}
        self._market_maker = market_maker

        # create a pool of predictors per asset
        self._predictors = {}
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

    def submit_orders(self, bitstring: np.ndarray, asset_indexes: dict):
        """
        Given the current world‐state bits, for each asset:
          1. Filter predictors whose condition string matches.
          2. Pick the lowest‐variance predictor.
          3. Compute CARA‐optimal holding: h* = (E[price+div] - p(1+r)) / (γ·Var)
          4. Round to integer shares, store in self._demand,
             and send (demand, slope=1/Var) to the market maker.
        """
        for asset, idx in asset_indexes.items():
            price    = self._market_maker.get_price(asset)
            dividend = self._market_maker.get_dividend(asset)

            # pick predictors that “fire” on the current bitstring
            active = [
                p for p in self._predictors[asset]
                if p.matches(bitstring[idx])
            ]
            if not active:
                # no signal ⇒ no position
                self._demand[asset] = 0
                continue

            # choose the most precise predictor
            best_var = min(p.get_variance() for p in active)
            best = next(p for p in active if p.get_variance() == best_var)

            # one‐step ahead forecast of total payout
            expected = best.predict(price, dividend)
            variance = best.get_variance()

            # CARA‐optimal target shares
            target_h = (expected - price * (1 + INTEREST_RATE)) / (
                RISK_AVERSION * variance
            )
            qty = int(np.round(target_h))

            # record for portfolio update
            self._demand[asset] = qty
            # submit to auction (slope = confidence = 1/variance)
            slope = 1.0 / (variance + 1e-8)
            self._market_maker.add_demand(asset, qty, slope)

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
