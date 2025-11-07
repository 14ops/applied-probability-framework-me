"""Factory helpers for creating broker interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Type

from .base_broker import BrokerInterface


class BrokerNotAvailableError(RuntimeError):
    """Raised when a requested broker cannot be instantiated."""


@dataclass
class _BrokerRegistration:
    name: str
    factory: Callable[..., BrokerInterface]
    display_name: str
    description: str
    supports_live: bool = True
    supports_paper: bool = True


class BrokerFactory:
    """Central registry for broker connectors."""

    _registry: Dict[str, _BrokerRegistration] = {}

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[..., BrokerInterface],
        display_name: Optional[str] = None,
        description: str = "",
        supports_live: bool = True,
        supports_paper: bool = True,
    ) -> None:
        cls._registry[name.lower()] = _BrokerRegistration(
            name=name.lower(),
            factory=factory,
            display_name=display_name or name,
            description=description,
            supports_live=supports_live,
            supports_paper=supports_paper,
        )

    @classmethod
    def create(cls, name: str, **kwargs) -> BrokerInterface:
        registration = cls._registry.get(name.lower())
        if not registration:
            raise BrokerNotAvailableError(f"Broker '{name}' is not registered")
        try:
            return registration.factory(**kwargs)
        except Exception as exc:  # pragma: no cover - depends on external SDKs
            raise BrokerNotAvailableError(str(exc)) from exc

    @classmethod
    def list_brokers(cls) -> Iterable[_BrokerRegistration]:
        return cls._registry.values()

    @classmethod
    def describe(cls, name: str) -> Optional[_BrokerRegistration]:
        return cls._registry.get(name.lower())


# Local import to avoid circular dependencies when the module is imported for
# registration side-effects.
def _auto_register() -> None:
    from .alpaca_interface import AlpacaInterface

    try:
        from .interactive_brokers_interface import InteractiveBrokersInterface
    except ImportError:  # Optional dependency, skip registration if missing
        InteractiveBrokersInterface = None  # type: ignore

    BrokerFactory.register(
        "alpaca",
        factory=lambda **kwargs: AlpacaInterface(**kwargs),
        display_name="Alpaca",
        description="Commission-free API broker supporting paper & live equities trading.",
        supports_live=True,
        supports_paper=True,
    )

    if InteractiveBrokersInterface is not None:
        BrokerFactory.register(
            "interactive_brokers",
            factory=lambda **kwargs: InteractiveBrokersInterface(**kwargs),
            display_name="Interactive Brokers",
            description="Professional-grade broker via TWS / IB Gateway.",
            supports_live=True,
            supports_paper=True,
        )


_auto_register()
