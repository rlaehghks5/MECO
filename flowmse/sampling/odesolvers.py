import abc
import torch
import numpy as np

from flowmse.util.registry import Registry


ODEsolverRegistry = Registry("ODEsolver")


class ODEsolver(abc.ABC):
    def __init__(self, ode, VF_fn):
        super().__init__()
        self.ode = ode
        self.VF_fn = VF_fn

    @abc.abstractmethod
    def update_fn(self, x, t, *args):
        pass


@ODEsolverRegistry.register('euler')
class EulerODEsolver(ODEsolver):
    def __init__(self, ode, VF_fn):
        super().__init__(ode, VF_fn)

    def update_fn(self, x, t, y, m, stepsize, *args):
        dt = -stepsize
        vectorfield = self.VF_fn(x, t, y, m) 
        x = x + vectorfield * dt
        return x


# -------------------- MeanFlow Euler --------------------
@ODEsolverRegistry.register('euler_mf')
class EulerMFODESolver(ODEsolver):
    def __init__(self, ode, VF_fn):
        super().__init__(ode, VF_fn)
        # print("euler_mf")
    def update_fn(self, x, t, y, m, stepsize, *args): # m: multi-channe mixture

        Delta = stepsize
        r = (t - Delta).clamp_min(0.0)
        u = self.VF_fn(x, t, y, m, r) # m: multi-channe mixture
        return x - Delta * u
# ----------------------------------------------------------------------
