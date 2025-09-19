from typing import List, Dict, Any


class SignatureDetector:
    """
    Handles signature detection using YOLO model
    """
    
    def __init__(self, config: ExtractorConfig):
        self.config = config
        self.model_manager = ModelManager(config)
    
    def detect_signatures(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect signatures in an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detection dictionaries
        """
        model = self.model_manager.get_model()
        
        # Run YOLO detection
        results = model.predict(
            source=image_path, 
            device=self.config.detection.device,
            verbose=False
        )
        
        # Process detections
        detections = self._process_yolo_results(results[0])
        
        # Apply selection logic
        detections = self._apply_selection_logic(detections)
        
        return detections
    
    def _process_yolo_results(self, results) -> List[Dict[str, Any]]:
        """Convert YOLO results to standardized format."""
        detections = []
        
        if getattr(results, "boxes", None) is None or len(results.boxes) == 0:
            return detections
        
        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2,
                "confidence": float(box.conf[0]),
                "class_id": int(box.cls[0])
            })
        
        return detections
    
    def _apply_selection_logic(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply confidence thresholds and selection rules."""
        conf_thresh = self.config.detection.conf_threshold
        ensure_one = self.config.detection.ensure_one_per_page
        
        # Check if any detection meets threshold
        any_likely = any(d["confidence"] >= conf_thresh for d in detections)
        best_idx = max(range(len(detections)), 
                      key=lambda i: detections[i]["confidence"]) if detections else None
        
        # Apply flags
        for i, detection in enumerate(detections):
            detection["likely_signature"] = detection["confidence"] >= conf_thresh
            detection["picked"] = detection["likely_signature"]
            detection["forced_pick"] = False
        
        # Force pick best if none meet threshold and ensure_one is True
        if ensure_one and not any_likely and best_idx is not None:
            detections[best_idx]["picked"] = True
            detections[best_idx]["forced_pick"] = True
        
        return detections
