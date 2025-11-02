"""
Registration script for character strategies.

This script registers all character strategies with the plugin system.
"""

from core.plugin_system import get_registry
from character_strategies import (
    TakeshiKovacsStrategy,
    LelouchViBritanniaStrategy,
    KazuyaKinoshitaStrategy,
    SenkuIshigamiStrategy,
    RintaroOkabeStrategy,
    HybridStrategy,
)


def register_all_character_strategies():
    """Register all character strategies with the global registry."""
    registry = get_registry()
    
    strategies = [
        ('takeshi', TakeshiKovacsStrategy),
        ('lelouch', LelouchViBritanniaStrategy),
        ('kazuya', KazuyaKinoshitaStrategy),
        ('senku', SenkuIshigamiStrategy),
        ('okabe', RintaroOkabeStrategy),
        ('hybrid', HybridStrategy),
    ]
    
    for name, strategy_class in strategies:
        try:
            registry.register_strategy(name, strategy_class)
            print(f"✓ Registered strategy: {name} ({strategy_class.__name__})")
        except ValueError as e:
            # Strategy already registered
            print(f"  Strategy '{name}' already registered: {e}")
        except Exception as e:
            print(f"✗ Failed to register strategy '{name}': {e}")
    
    print(f"\nTotal strategies registered: {len(registry.list_strategies())}")
    print(f"Available strategies: {', '.join(registry.list_strategies())}")


if __name__ == "__main__":
    register_all_character_strategies()














