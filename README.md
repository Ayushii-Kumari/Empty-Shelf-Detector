
# 🛒 Empty Shelf Detector

An automated inventory monitoring system that detects empty spaces on retail shelves in real-time. This project uses **Computer Vision (YOLOv8)** to identify gaps and provides a user-friendly interface via **Gradio**.

---

## 📺 Project Demo



https://github.com/user-attachments/assets/95ae97fa-dc60-4923-9f34-600f7b3fb0c8


---

## 🚀 Key Features

* **Real-time Detection:** Identifies empty shelf slots in images and video streams.
* **YOLOv8 Powered:** Built on the state-of-the-art Ultralytics YOLOv8 architecture for high accuracy and speed.
* **Interactive UI:** Simple drag-and-drop interface for users to test their own shelf images.
* **Custom Training:** Includes logic for training on specific retail datasets.

---

## 📂 Project Structure

```text
├── .gradio/               # Cached UI files
├── myenv/                 # Virtual environment (Local only)
├── runs/                  # YOLOv8 training results and weights
├── data.yaml              # Dataset configuration for YOLO
├── Empty Shelf Detector.py # Main Gradio application script
├── model.py               # Model architecture and loading logic
├── testing.py             # Script for batch testing images
├── yolov8n.pt             # Pre-trained YOLOv8 weights
└── requirements.txt       # Necessary Python libraries

```

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Ayushii-Kumari/Empty-Shelf-Detector.git
cd Empty-Shelf-Detector

```


2. **Create and activate a virtual environment:**
```bash
python -m venv myenv
# On Windows:
myenv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

## 💻 Usage

To launch the detection interface, run:

```bash
python "Empty Shelf Detector.py"

```

Once running, a local URL (e.g., `http://127.0.0.1:7860`) will appear in your terminal. Open it in your browser to start detecting!

---

## 📊 Model Training

The model was trained using the `data.yaml` configuration.

* **Base Model:** YOLOv8 Nano
* **Input Size:** 640px
* **Output:** Detection of "Empty" vs "Filled" shelf slots.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
