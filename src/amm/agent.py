from abc import ABC, abstractmethod
from data import Action

class Agent(ABC):
    def __init__(self, id: str, cash: float):
        self._id = id
        self._position = None
        self._cash = cash

    @abstractmethod
    def act(self) -> Action:
        pass
