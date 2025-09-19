from typing import Optional
import time


class ProgressTracker:
    """
    Tracks and logs processing progress.
    
    Single Responsibility: Progress monitoring and reporting.
    """
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.description = description
        self.current_item = 0
        self.start_time = time.time()
        self.errors = 0
    
    def update(self, increment: int = 1, error: bool = False):
        """Update progress counter."""
        self.current_item += increment
        if error:
            self.errors += 1
        
        # Log progress every 10 items or at completion
        if self.current_item % 10 == 0 or self.current_item == self.total_items:
            self._log_progress()
    
    def _log_progress(self):
        """Log current progress."""
        percentage = (self.current_item / self.total_items) * 100
        elapsed = time.time() - self.start_time
        rate = self.current_item / elapsed if elapsed > 0 else 0
        
        logger.info(
            f"{self.description}: {self.current_item}/{self.total_items} "
            f"({percentage:.1f}%) - {rate:.1f} items/sec - {self.errors} errors"
        )
