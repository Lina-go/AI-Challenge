from .config import (
    ExtractorConfig,
    DetectionConfig, 
    ProcessingConfig
)

from .base_extractor import (
    BaseSourceAdapter,
    BaseExtractor
)

from .adapters import (
    CSVSourceAdapter,
    DirectorySourceAdapter
)

from .core import (
    ExtractionOrchestrator,
    PDFProcessor,
    ModelManager,
    SignatureDetector,
    ImageProcessor,
    ResultManager
)

__version__ = "1.0.0"
__all__ = [
    # Configuration
    "ExtractorConfig",
    "DetectionConfig", 
    "ProcessingConfig",
    
    # Base classes
    "BaseSourceAdapter",
    "BaseExtractor",
    
    # Adapters
    "CSVSourceAdapter",
    "DirectorySourceAdapter",
    
    # Core components
    "ExtractionOrchestrator",
    "PDFProcessor",
    "ModelManager", 
    "SignatureDetector",
    "ImageProcessor",
    "ResultManager"
]