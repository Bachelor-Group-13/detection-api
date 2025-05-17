# Inneparkert Detection API

The detection service for Inneparkert, a smart parking management system developed as a bachelor project in collaboration with Twoday.

## Overview

This is the detection service of the Inneparkert system, built with Python and FastAPI. It provides real-time vehicle and license plate detection capabilities, serving as a crucial component in the parking management system.

## Features

- **Vehicle Detection**
  - Real-time vehicle detection using YOLOv8
  - Position classification (front/back)
  - Confidence scoring
  - Visual bounding box detection

- **License Plate Recognition**
  - Automatic license plate detection
  - OCR-based plate number recognition
  - Norwegian license plate format validation

## Technical Details

### Built With

- **Framework**: FastAPI
- **Language**: Python
- **ML Models**: 
  - YOLOv8 for object detection
  - EasyOCR for text recognition
- **Image Processing**: OpenCV

### Project Structure

python-api/
├── routes/
│ ├── parking_detection.py
│ └── license_plate.py
├── utils/
│ └── image_processing.py
├── main.py
├── requirements.txt
└── Dockerfile


## API Endpoints

- `POST /parking-detection`
  - Detects vehicles in parking areas
  - Returns vehicle positions and processed image

- `POST /license-plate`
  - Detects and recognizes license plates
  - Returns list of valid plate numbers

## Development Setup

### Prerequisites

- Python 3.8+
- Docker (optional)

### Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

### Docker

Build and run with Docker:
```bash
docker build -t inneparkert-detection .
docker run -p 8000:8000 inneparkert-detection
```

## Project Status

This service is part of the Inneparkert system and provides real-time detection capabilities for the main application.

## Team

- Viljar Hoem-Olsen
- Thomas Åkre
- Sander Grimstad
