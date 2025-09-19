from abc import ABC, abstractmethod
from typing import List, Dict, Any
import polars as pl


class BaseSourceAdapter(ABC):
    """Abstract base class for different PDF sources."""
    
    def __init__(self, config: ExtractorConfig):
        self.config = config
    
    @abstractmethod
    def get_pdf_sources(self) -> List[Dict[str, Any]]:
        """
        Get list of PDF sources to process.
        
        Returns:
            List of dicts with keys: 'source_id', 'pdf_path', 'metadata'
        """
        pass
        
    @abstractmethod
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """
        Prepare a source for processing (e.g., download if URL).
        
        Returns:
            Local path to PDF file
        """
        pass


class BaseExtractor(ABC):
    """Abstract base class for signature extraction workflows."""
    
    def __init__(self, config: ExtractorConfig):
        self.config = config
        self._setup_components()
    
    @abstractmethod
    def _setup_components(self):
        """Initialize extraction components."""
        pass
    
    @abstractmethod
    def extract_signatures(self, sources: List[Dict[str, Any]]) -> pl.DataFrame:
        """Extract signatures from PDF sources."""
        pass