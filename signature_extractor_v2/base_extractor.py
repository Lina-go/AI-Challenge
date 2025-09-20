from abc import ABC, abstractmethod
from typing import List, Dict, Any
from .config import ExtractionConfig

class BaseSourceAdapter(ABC):
    """Abstract base class for different PDF sources"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
    
    @abstractmethod
    def get_pdf_sources(self) -> List[Dict[str, Any]]:
        """Get list of PDF sources to process"""
        pass
        
    @abstractmethod
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """Prepare source for processing (download if needed)"""
        pass

class BaseExtractor(ABC):
    """Abstract base class for signature extraction workflows"""
    
    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._setup_components()
    
    @abstractmethod
    def _setup_components(self):
        """Initialize extraction components"""
        pass
    
    @abstractmethod
    def extract_signatures(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract signatures from PDF sources"""
        pass