"""Broker API package."""

from .broker_clients import BrokerClient, BrokerDefinition, BrokerError, REGISTRY

# Import broker implementations to ensure they register themselves
from . import alpaca_broker  # noqa: F401
from . import manual_broker  # noqa: F401

__all__ = [
    "BrokerClient",
    "BrokerDefinition",
    "BrokerError",
    "REGISTRY",
]
