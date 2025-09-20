from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseSourceAdapter(ABC):
    """Abstract base class for different PDF sources"""
    
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def get_pdf_sources(self) -> List[Dict[str, Any]]:
        """Get list of PDF sources to process"""
        pass
        
    @abstractmethod
    def prepare_source(self, source: Dict[str, Any]) -> str:
        """Prepare source for processing (download if needed)"""
        pass