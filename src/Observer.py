from abc import ABC, abstractmethod

class Subject: pass

class Observer(ABC):
    """description of class"""

    @abstractmethod
    def Update(self, changedSubject: Subject): 
        pass



