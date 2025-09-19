"""
Site-specific adapters for different modern slavery statement repositories.
"""

from .canadian import CanadianAdapter
from .australian import AustralianAdapter

__all__ = ["CanadianAdapter", "AustralianAdapter"]
