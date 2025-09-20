# signature_extractor_v5/adapters/base.py
"""Base adapter interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseSourceAdapter(ABC):
    """Abstract base class for source adapters."""
    
    @abstractmethod
    def get_sources(self) -> List[Dict[str, Any]]:
        """Get list of PDF sources."""
        pass
    
    @abstractmethod
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """Prepare source for processing."""
        pass