# ✏️ Handwritten Digit Recognition with Back-Propagation Network

This repository contains a replication of the classic paper **"Handwritten Digit Recognition with a Back-Propagation Network"** by LeCun et al., AT&T Bell Laboratories. The goal is to implement the network as described in the paper, using minimal preprocessing and a constrained architecture to perform handwritten digit recognition.

**Paper**: [Handwritten Digit Recognition with a Back-Propagation Network (NeurIPS 1989)](https://proceedings.neurips.cc/paper_files/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf)

---

## 🖼 About the Dataset

The original USPS digit dataset was used in the paper. However, there were issues downloading and processing the raw USPS data. To make things easier, a preprocessed HDF5 file (`usps.h5`) is provided, containing all training and testing images and labels in a ready-to-use format.  

Images have been normalized to `[-1, 1]` and resized to `16x16` pixels following the preprocessing described in the paper.

---

## 🏗️ Model Architecture

The network architecture follows the design described in **Figure 4** of the original paper.  

![Handwritten Digit Recognition Model](images/fig4.png)

The figure illustrates the fully connected back-propagation network used for digit recognition, showing the input layer, hidden layers, and output layer with 10 neurons corresponding to the digits 0-9.

---

## 🗂 Project Structure

```bash

HandwrittenDigitRecognition-BP-PaperReplicating/
│
├── data/
│   ├── train/
│   ├── test/
│   └── usps.h5
│
├── dataproc/
│   ├── preprocess.py
│   └── preview.py
│
├── models/
│   ├── bp_network.py
│   └── layers.py
│
├── utils/
│   ├── dataset.py
│   ├── metrics.py
│   └── visualization.py
│
├── requirements.txt
└── README.md


```
---
## 🔗 Feedback

For feedback or questions, contact: [barkin.adiguzel@gmail.com](mailto:barkin.adiguzel@gmail.com)
