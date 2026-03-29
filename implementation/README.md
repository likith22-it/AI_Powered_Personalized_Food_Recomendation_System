# Food Weight Estimator (Flask)

This repository contains a simple Flask web app that:

- Detects food items in an image using YOLOv8 (`ultralytics`)
- Estimates depth using MiDaS
- Classifies each detected food item using a ViT model
- Estimates grams based on a simple depth/area/density heuristic

## Setup

1. Create a Python environment (recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your model files in a local `models/` directory (or set environment variables):

- `models/best.pt` (YOLO weights)
- `models/vit_tiny_food_models.pth` (ViT classifier weights)

You can override the paths with environment variables:

- `YOLO_WEIGHTS`
- `VIT_WEIGHTS`

## Run

```bash
python app.py
```

Open your browser at: http://localhost:5000

## Notes

- The first request may take a while because the models are loaded into memory.
- The weight estimate is very rough since it uses a simple depth/area heuristic and fixed densities.
