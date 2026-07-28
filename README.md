#  PerceptionLab

A modular computer vision framework for unified scene understanding using state-of-the-art foundation models.

PerceptionLab combines open-vocabulary object detection, promptable segmentation, monocular depth estimation, and an interactive visualization interface into a single perception pipeline.

---


## Features

- Automatic object proposal generation using Florence-2
- Open-Vocabulary Object Detection using Grounding DINO
- Automatic Object Prompt Generation using Florence-2
- Image Captioning with Florence-2
- Zero-shot Instance Segmentation with SAM2.1
- Monocular Depth Estimation with Depth Anything V2
- Scene Graph Generation
- Visual Question Answering
- Unified Benchmark Dashboard
- Interactive Gradio Interface

---

## Current Pipeline

```
                +----------------------+
                |      Input Image     |
                +----------+-----------+
                           |
        +------------------+------------------+
        |                  |                  |
        v                  v                  v
+----------------+  +---------------+  +----------------+
| Florence-2     |  | Grounding DINO|  | Depth Anything |
| Captioning     |  | Open-Vocabulary| | V2             |
+--------+-------+  | Detection      | +----------------+
         |          +-------+--------+
         |                  |
         |        Florence-2 Object Proposal
         |                  |
         +-----------------+
                           |
                           v
                    Grounding DINO
                           |
                           v
                         SAM2.1
                           |
                           v
                Segmentation + Visualization
                           |
                           v
               Scene Graph + Visual Question Answering
```

---

## Current Modules

## Models

| Module | Model |
|---------|-------|
| Captioning | Microsoft Florence-2 Base |
| Object Proposal | Microsoft Florence-2 Base |
| Detection | Grounding DINO |
| Segmentation | SAM2.1 Hiera Base+ |
| Depth Estimation | Depth Anything V2 Base |
| Scene Graph | Custom |
| VQA | LLaVA / Custom |

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
|   ├── florence2.py
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

- Florence-2 Captioning
- Florence-2 Automatic Object Proposal
- Depth Anything V2
- Benchmark Dashboard
- Unified Visualization

###  Planned

- Scene Graph Generation
- Visual Question Answering
- OCR
- Video Support
- Multi-image Reasoning
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
