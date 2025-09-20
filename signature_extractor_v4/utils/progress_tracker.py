import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Progress tracking for processing operations"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = time.time()
        self.errors = 0
        self.last_update = self.start_time
    
    def update(self, increment: int = 1, error: bool = False):
        """Update progress with metrics"""
        self.current_item += increment
        if error:
            self.errors += 1
        
        current_time = time.time()
        
        # Log every 10 items or significant milestones
        if (self.current_item % 10 == 0 or 
            self.current_item == self.total_items or
            current_time - self.last_update > 30):
            
            self._log_progress()
            self.last_update = current_time
    
    def _log_progress(self):
        """Log detailed progress information"""
        elapsed = time.time() - self.start_time
        percentage = (self.current_item / self.total_items) * 100
        rate = self.current_item / elapsed if elapsed > 0 else 0
        
        # Calculate ETA
        if rate > 0:
            remaining_items = self.total_items - self.current_item
            eta_seconds = remaining_items / rate
            eta_str = f"ETA: {eta_seconds/60:.1f}m"
        else:
            eta_str = "ETA: calculating..."
        
        logger.info(
            f"{self.description}: {self.current_item}/{self.total_items} "
            f"({percentage:.1f}%) - {rate:.2f} items/sec - "
            f"{self.errors} errors - {eta_str}"
        )