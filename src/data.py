from dataclasses import dataclass
from enum import Enum

QUOTE_ASSET = "USD"

class OrderType(Enum):
    BUY = "buy"
    SELL = "sell"

class ActionSide(Enum):
    BUY = 1
    HOLD = 0
    SELL = -1

@dataclass
class Action:
    side: ActionSide
    asset: str