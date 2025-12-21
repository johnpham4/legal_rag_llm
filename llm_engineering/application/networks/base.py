from typing import ClassVar
from threading import Lock
from abc import ABCMeta


class SingletonMeta(ABCMeta):
    _instances: ClassVar = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwds):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwds)
                cls._instances[cls] = instance
        return cls._instances[cls]


