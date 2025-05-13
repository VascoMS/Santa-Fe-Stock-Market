class Pool:
    def __init__(self, token_a: str, token_b: str, fee: float):
        self.token_a = token_a
        self.token_b = token_b
        self.fee = fee
        self.reserve_a = 0
        self.reserve_b = 0
        self.fees_collected = 0
    
    def get_price(self) -> float:    
        if self.reserve_a == 0 or self.reserve_b == 0:
            return 0
        return self.reserve_a / self.reserve_b

    def add_liquidity(self, amount_a: float, amount_b: float) -> None:
        self.reserve_a += amount_a
        self.reserve_b += amount_b
        self.latest_k = self.reserve_a * self.reserve_b
    
    def swap(self, amount_in: float, token_in: str) -> float:
        amount_in = amount_in * (1 - self.fee)
        self.fees_collected += amount_in * self.fee
        if token_in == self.token_a:
            amount_out = (self.reserve_b * amount_in) / (self.reserve_a + amount_in)
            self.reserve_a += amount_in
            self.reserve_b -= amount_out
        elif token_in == self.token_b:
            amount_out = (self.reserve_a * amount_in) / (self.reserve_b + amount_in)
            self.reserve_b += amount_in
            self.reserve_a -= amount_out
        else:
            raise ValueError(f"Invalid token {token_in}")
        return amount_out