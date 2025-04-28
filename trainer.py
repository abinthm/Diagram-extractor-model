import os
import cv2
from ultralytics import YOLO
import supervision as sv
from roboflow import Roboflow

class DiagramExtractor:
    def __init__(self, model_path=None, confidence=0.5):
        """
        Initialize Diagram Extractor
        
        Args:
            model_path (str): Path to trained or pre-trained model
            confidence (float): Confidence threshold for detection
        """
        # Create output directories
        self.output_dir = 'extracted_diagrams'
        os.makedirs(self.output_dir, exist_ok=True)
    

                
        # Download dataset using Roboflow
        rf = Roboflow(api_key="9ln8sWXSPXpef56x38Ow")
        project = rf.workspace("diagram-extraction").project("diagram-extractor")
        version = project.version(1)
        dataset = version.download("yolov8")
        
        # Path to the downloaded dataset's data.yaml
        self.data_yaml = os.path.join(dataset.location, "data.yaml")
        
        # Load model
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Train a new model or use pre-trained
            self.model = self.train_model()
        
        self.confidence = confidence
    
    def train_model(self):
        """
        Train YOLO model on custom dataset
        
        Returns:
            Trained YOLO model
        """
        try:
            # Start with pre-trained weights
            model = YOLO('yolov8n.pt')
            
            # Training configuration
            results = model.train(
                data=self.data_yaml,
                epochs=100,
                imgsz=640,
                batch=16,
                patience=10,
                plots=True
            )
            
            # Save the trained model
            best_model_path = 'runs/detect/train/weights/best.pt'
            if os.path.exists(best_model_path):
                return YOLO(best_model_path)
            return model
        except Exception as e:
            print(f"Training error: {e}")
            raise
    
    def detect_diagrams(self, image_path):
        """
        Detect and extract diagrams from an image
        
        Args:
            image_path (str): Path to input image
        
        Returns:
            List of extracted diagram paths
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            
            # Perform detection
            results = self.model(image, conf=self.confidence)
            
            extracted_diagrams = []
            
            # Process each detected diagram
            for i, result in enumerate(results):
                boxes = result.boxes
                
                for j, box in enumerate(boxes):
                    # Extract coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Extract diagram
                    diagram = image[y1:y2, x1:x2]
                    
                    # Save diagram
                    diagram_path = os.path.join(
                        self.output_dir, 
                        f'diagram_{os.path.basename(image_path)}_{i}_{j}.png'
                    )
                    cv2.imwrite(diagram_path, diagram)
                    extracted_diagrams.append(diagram_path)
                    
                    # Visualize detection (optional)
                    self._visualize_detection(image, box)
            
            return extracted_diagrams
        
        except Exception as e:
            print(f"Diagram detection error: {e}")
            return []
    
    def _visualize_detection(self, image, box):
        """
        Visualize detection results
        
        Args:
            image (np.ndarray): Original image
            box (torch.Tensor): Detected box
        """
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    def batch_process(self, image_folder):
        """
        Process multiple images in a folder
        
        Args:
            image_folder (str): Path to folder with images
        
        Returns:
            List of all extracted diagram paths
        """
        all_diagrams = []
        
        # Supported image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(image_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(image_folder, filename)
                diagrams = self.detect_diagrams(image_path)
                all_diagrams.extend(diagrams)
        
        return all_diagrams

def main():
    try:
        # Initialize extractor
        extractor = DiagramExtractor()
        
        # Single image processing
        # Replace with path to your test image
        test_image_path = 'D:\YOLOX\CN\page4_837.jpg'
        diagrams = extractor.detect_diagrams(test_image_path)
        print(f"Extracted {len(diagrams)} diagrams")
        
        # Optional: Batch processing
        # Replace with path to your image folder
        # all_diagrams = extractor.batch_process('path/to/image/folder')
    
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == '__main__':
    main()