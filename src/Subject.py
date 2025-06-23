from abc import ABC, abstractmethod
from Observer import Observer

class Subject(ABC):
    """description of class"""

    def __init__(self):
        self.__obs = []


    def Attach(self, observer: Observer) -> None:
        self.__obs.append(observer)

    def Detach(self, observer: Observer) -> None:
        self.__obs.remove(observer);

    def Notify(self) -> None:
        for observer in self.__obs:
            observer.Update(self)