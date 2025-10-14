from __future__ import annotations

from ..backends import backends, DriverBase

import os

def _create_driver() -> DriverBase:
    if os.getenv("TRITON_CPU_BACKEND", "0") == "1":
        if "cpu" not in backends:
            raise RuntimeError("TRITON_CPU_BACKEND is set, but CPU backend is unavailable.")
        return backends["cpu"].driver()

    actives = [x.driver for x in backends.values() if x.driver.is_active()]
    if len(actives) >= 2 and backends["cpu"].driver.is_active():
        actives.remove(backends["cpu"].driver)
    if len(actives) != 1:
        raise RuntimeError(f"{len(actives)} active drivers ({actives}). There should only be one.")
    return actives[0]()


class DriverConfig:

    def __init__(self) -> None:
        self._default: DriverBase | None = None
        self._active: DriverBase | None = None

    @property
    def default(self) -> DriverBase:
        if self._default is None:
            self._default = _create_driver()
        return self._default

    @property
    def active(self) -> DriverBase:
        if self._active is None:
            self._active = self.default
        return self._active

    def set_active(self, driver: DriverBase) -> None:
        self._active = driver

    def reset_active(self) -> None:
        self._active = self.default

    def set_active_to_cpu(self):
        if "cpu" not in backends:
            raise RuntimeError("CPU backend is unavailable")
        self.active = backends["cpu"].driver()

    def set_active_to_gpu(self):
        active_gpus = [(name, backend.driver)
                       for name, backend in backends.items()
                       if backend.driver.is_active() and name != "cpu"]
        if len(active_gpus) != 1:
            raise RuntimeError(f"{len(active_gpus)} active GPU drivers ({active_gpus}). There should only be one GPU.")
        self.active = active_gpus[0][1]()
        return active_gpus[0][0]

    def get_active_gpus(self):
        return [name for name, backend in backends.items() if backend.driver.is_active() and name != "cpu"]


driver = DriverConfig()
