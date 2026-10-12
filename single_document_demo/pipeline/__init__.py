"""Public configuration and execution API for the two demo workflows."""

from .config import DemoConfig, SUPPORTED_WORKFLOWS, load_demo_config
from .runner import resolve_run_mode, run_demo

__all__ = [
    "DemoConfig",
    "SUPPORTED_WORKFLOWS",
    "load_demo_config",
    "resolve_run_mode",
    "run_demo",
]
