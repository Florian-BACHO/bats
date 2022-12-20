from typing import Any

from bats.AbstractMonitor import AbstractMonitor
import numpy as np

class ValueMonitor(AbstractMonitor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__value: float = 0.0

    def add(self, value: float):
        self.__value = value

    def record(self, epoch) -> None:
        self._record(epoch, self.__value)
