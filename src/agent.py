class Agent:
    def __init__(self, id: str, cash: float):
        self._id = id
        self._portfolio = {"asset_1": 0, "asset_2": 0, "asset_3": 0}
        self._cash = cash
        self._demand = {"asset_1": 0, "asset_2": 0, "asset_3": 0}

