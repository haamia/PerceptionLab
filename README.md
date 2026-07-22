# 🔬 PerceptionLab

A modular computer vision framework for unified scene understanding using state-of-the-art foundation models.

PerceptionLab combines open-vocabulary object detection, promptable segmentation, monocular depth estimation, and an interactive visualization interface into a single perception pipeline.

---

## ✨ Features

-  Open Vocabulary Object Detection (Grounding DINO)
-  Promptable Segmentation (SAM2)
-  Monocular Depth Estimation (Depth Anything V2)
-  Performance Benchmark Dashboard
-  Unified Visualization
-  Interactive Gradio Interface
-  Modular Architecture

---

## Current Pipeline

```
                Input Image
                     │
                     ▼
            Grounding DINO
                     │
             DetectionResult
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       SAM2            Depth Anything V2
          │                     │
SegmentationResult      DepthResult
          └──────────┬──────────┘
                     ▼
          PerceptionVisualizer
                     │
                     ▼
               Final Output
```

---

## Current Modules

| Module | Status |
|---------|:------:|
| Grounding DINO | ✅ |
| SAM2 | ✅ |
| Depth Anything V2 | ✅ |
| Detection Visualization | ✅ |
| Segmentation Visualization | ✅ |
| Depth Visualization | ✅ |
| Benchmark Dashboard | ✅ |
| Gradio UI | ✅ |

---

## Project Structure

```
PerceptionLab/

├── app.py
├── config.py
│
├── core/
│   ├── managers/
│   └── results.py
│
├── models/
│   ├── grounding_dino.py
│   ├── sam2.py
│   ├── depth_anything.py
│   └── base_model.py
│
├── services/
│   └── pipeline.py
│
├── ui/
│   └── dashboard.py
│
├── utils/
│   ├── logger.py
│   └── visualization/
│       ├── colors.py
│       ├── detection.py
│       ├── segmentation.py
│       ├── depth.py
│       └── perception.py
```

---

## Technologies

- Python
- PyTorch
- Transformers
- Grounding DINO
- SAM2
- Depth Anything V2
- OpenCV
- Gradio

---

## Installation

```bash
git clone https://github.com/haamia/PerceptionLab.git

cd PerceptionLab

python -m venv venv

source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

Run

```bash
python app.py
```

---

## Roadmap

###  v0.1

- Grounding DINO
- SAM2

###  v0.2

- Depth Anything V2
- Unified Visualization
- Benchmark Dashboard

###  Planned

- Florence-2 Captioning
- Scene Graph Generation
- Visual Question Answering
- OCR
- Multi-image Reasoning
- Video Support
- Model Zoo
- Fast / High Accuracy Modes

---

## License

MIT License

---

## Author

**Haamia Farooq**

Department of Mechatronics Engineering

National University of Sciences and Technology (NUST)
