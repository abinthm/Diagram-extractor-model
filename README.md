# Diagram Extraction Model

This project implements a diagram extraction system using YOLOv8 for detecting and extracting diagrams from images. It's built using the Ultralytics YOLO framework and includes features for both single image and batch processing.

## Features

- Diagram detection using YOLOv8
- Single image and batch processing capabilities
- Automatic dataset download from Roboflow
- Visualization of detection results
- Extracted diagrams are saved in a dedicated output directory

## Prerequisites

- NVIDIA GPU with CUDA support (recommended)
- Conda package manager

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate diagram-extraction-model
```

## Usage

### Single Image Processing

```python
from trainer import DiagramExtractor

# Initialize the extractor
extractor = DiagramExtractor()

# Process a single image
diagrams = extractor.detect_diagrams('path/to/your/image.jpg')
print(f"Extracted {len(diagrams)} diagrams")
```

### Batch Processing

```python
# Process all images in a folder
all_diagrams = extractor.batch_process('path/to/image/folder')
```

## Configuration

The `DiagramExtractor` class accepts the following parameters:
- `model_path`: Path to a pre-trained model (optional)
- `confidence`: Confidence threshold for detection (default: 0.5)

## Output

Extracted diagrams are saved in the `extracted_diagrams` directory with filenames following the pattern:
`diagram_[original_filename]_[detection_index]_[box_index].png`

## License

[Add your license information here]

## Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com)
- [Supervision](https://github.com/roboflow/supervision) 