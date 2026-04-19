# Microseismic Classification

> Classification AI to determine where seismic events are given a SEGY file.

Using data from DAS sensors collecting seismic data, this project finds microseismic events. It uses a DSP with raking, normilization and Sobel edge detection, then uses a ResNet CNN to do a binary classification on chunked segments of the data. This data comes from the Univeristy of Utah Forge project.

## Screenshots

![Alt text](/screenshots/web-example.jpg?raw=true "Web")

## Features

- Binary classification
- Dataset preprocessing
- Web interface

## Dependencies

- openpyxl
- matplotlib
- numpy
- segyio
- pandas
- torch
- torchvision
- scikit-learn
- flask / flask-cors
- PIL
- h5py
- python-dotenv

```bash
pip install openpyxl matplotlib numpy segyio pandas torch torchvision scikit-learn python-dotenv flask flask-cors Pillow h5py
```

## Installation

### Clone

```bash
git clone https://github.com/AurickMikleton/MicroSeismic.git
```

### Run

```bash
python app.py
```

## Presentaiton

<a href="https://canva.link/3hzu9q121mmmj4e">Presentaion Link</a>
