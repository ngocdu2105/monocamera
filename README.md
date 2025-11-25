# M-Calib: Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System



## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)

## Introduction
Find the coordinates of the center point of the object on the bottom surface. Then convert to the chessboard coordinates corresponding to the reference point.
## Our proposed:
<img src="https://github.com/thanhnguyencanh/MonoCalibNet/blob/main/image/Overview.png" width="750px">

## Installation

1. **Clone the Repository**:
   ```bash
   
   git clone https://github.com/ngocdu2105/monocamera.git
   cd monocamera
   
2. **Create virtual environment and install**:
   ```bash
   
   pip install -r requirements.txt

3. **Run**:
   ```bash
   
   python -m src.python.main
## Usage

#### Download ONNX Models and Dataset

Download the ONNX models YOLOv5, RCNN into the `models` directory. And the image dataset from the following Google Drive into `dataset/img`.

**Google Drive Link**: [Download](https://drive.google.com/drive/folders/1y-XrTXRQywmW5O1Tz1JYiJnR0lOq0Iyn?hl=vi)

## Demo
The results obtained are demonstrated as shown in the image below. 
<p align="center">
  <img src="demo/demo.png" alt="Alt text" />
</p>




Implementation code for our paper:  
["M-Calib: A Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System"](https://assets.researchsquare.com/files/rs-4019542/v1_covered_5a75ac68-1bc8-4bdd-b2c5-8bbdb1eac8f1.pdf?c=1711473654)

**Authors:** Thanh Nguyen Canh, Du Trinh Ngoc, Xiem HoangVan  
**Journal:** *Journal of Automation, Mobile Robotics and Intelligent Systems*, 2024  

---

## Citation
```bibtex
@article{canhmonocular2025,
  title = {Monocular 3D Object Localization using 2D Estimates for Industrial Robot Vision System},
  url = {https://www.jamris.org/index.php/JAMRIS/article/view/1485},
  DOI = {https://doi.org/10.14313/jamris-2025-025},
  journal = {Journal of Automation, Mobile Robotics and Intelligent Systems},
  publisher = {Industrial Research Institute for Automation and Measurements PIAP, Poland},
  author = {Thanh, Nguyen Canh and Du, Trinh Ngoc and Xiem HoangVan},
  year = {2025},
  month = sep,
  volume = {19},
  pages = {53--65}
}

