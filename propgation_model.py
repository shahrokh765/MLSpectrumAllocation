import abc
from Commons.elements import Element
from Commons.Point import Point
from math import log10
from random import gauss


class PropagationModel(abc.ABC):
    def __init__(self, name: str):
        self.__name = name

    @property
    def name(self) -> str:
        return self.__name

    @abc.abstractmethod
    def path_loss(self, tx: Element, rx: Element, iteration: int=0) -> float:
        """Return path loss from source to destination in dB"""
        pass

    @abc.abstractmethod
    def __str__(self):
        pass


class LogDistancePM(PropagationModel):
    def __init__(self, alpha: float, noise: bool=False, std: float=0.0, pl_ref: float=0.0, dist_ref: float=1.0):
        """Log-Distance(Normal) propagation model with path loss coefficient alpha and a standard deviation if noise
        is allowed."""
        PropagationModel.__init__(self, 'log')
        self.__alpha = alpha        # path loss coefficient
        self.__noise = noise        # indicates if there is noise or not
        self.__std = std            # standard deviation of the noise with mean of zero
        self.__pl_ref = pl_ref      # Path loss reference (PL_0) in distance distance reference dist_ref
        self.__dist_ref = dist_ref  # Distance reference where the path loss is pl_ref

    @property
    def alpha(self) -> float:
        """:return: return path-loss coefficient"""
        return self.__alpha

    @alpha.setter
    def alpha(self, alpha: float):
        """:param: set path-loss coefficient(alpha)"""
        self.__alpha = alpha

    @property
    def is_noisy(self) -> bool:
        return self.__noise

    @is_noisy.setter
    def is_noisy(self, noise: bool):
        self.__noise = noise

    @property
    def std(self) -> float:
        return self.__std

    @std.setter
    def std(self, std: float):
        self.__std = std

    @property
    def pl_ref(self) -> float:
        """:return: return path-loss reference that defines path loss in dist_ref distance."""
        return self.__pl_ref

    @pl_ref.setter
    def pl_ref(self, pl_ref: bool):
        """:param: set path-loss reference that defines path loss in dist_ref distance."""
        self.__pl_ref = pl_ref

    @property
    def dist_ref(self) -> float:
        """:return: return distance reference where the path loss is pl_ref."""
        return self.__dist_ref

    @dist_ref.setter
    def dist_ref(self, dist_ref: float):
        """:param: set distance reference where the path loss is pl_ref."""
        self.__dist_ref = dist_ref

    def path_loss(self, tx: Element, rx: Element, iteration: int=0) -> float:
        distance = tx.location.distance(rx.location) / self.dist_ref
        loss = self.pl_ref
        loss += 0 if distance < 1 else 10 * self.alpha * log10(distance)
        if self.is_noisy:
            loss += gauss(0, self.std)
        return loss

    def __str__(self):
        return "Log-Distance(Normal):\n" +\
               "Path-Loss Coeff.= {0}\n".format(self.alpha) +\
               "Path-Loss reference {0} in distance reference {1}\n".format(self.pl_ref, self.dist_ref) +\
               ("std= {0}" if self.is_noisy else "").format(self.std if self.is_noisy else None)


if __name__ == "__main__":
    log_pm = LogDistancePM(2.0, True, 1.0)
    print(log_pm)
    print(log_pm.path_loss(Element(Point((0, 0)), 15), Element(Point((10, 10)), 15)))
    log_pm.is_noisy = False
    print(log_pm)
    print(hasattr(log_pm, 'alpha'))