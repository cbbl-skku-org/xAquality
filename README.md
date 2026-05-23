# xAquality

![xAquality Architecture](figures/xAquality.png)

## Abstract
Coming soon...

## Online Demo (Hugging Face Spaces)
You can try the live demo of the application on Hugging Face Spaces here:
[https://huggingface.co/spaces/cbbl-skku-org/xAquality](https://huggingface.co/spaces/cbbl-skku-org/xAquality)

> **Note:** If the Space is currently "sleeping" due to inactivity, simply click the "Restart this Space" button and wait a few moments for it to wake up!

## Getting Started

### Preparing the Environment
To run this project locally, first clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/nhattruongpham/xAquality.git
cd xAquality
pip install -r requirements.txt
```
*(Tip: We recommend using a virtual environment like `conda` or `venv` to keep dependencies isolated.)*

## Running the Application

### 1. Web Dashboard (Gradio)
To launch the interactive dashboard for Single Sample and Batch Predictions:
```bash
python app.py
```
This will start a local web server (usually at `http://127.0.0.1:7860/`), where you can interact with the model via your web browser.

### 2. Command-Line Inference
To run predictions directly from the command line on a CSV dataset, use the `inference.py` script:
```bash
python inference.py
```
Make sure you configure the script to point to your specific input CSV file.

## About Us
Visit the **Computational Biology and Bioinformatics Laboratory (CBBL)** at Sungkyunkwan University (SKKU) to learn more about our research: [https://balalab-skku.org/](https://balalab-skku.org/)

## Citation

If you find this repository useful, please cite our work:

```bibtex
Coming soon...
```