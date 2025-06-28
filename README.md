# EEG Emotion Recognition using SEED-V Dataset

This project focuses on classifying human emotions from EEG brain signals using machine learning and deep learning techniques.  
Raw EEG data is processed from `.cnt` files using the MNE library, followed by signal filtering, ICA, and CSV export.  
The final models are trained on this data to predict emotional states.

## Project Structure

emotion-recognition/
├── src/
│ ├── preprocessing/ # EEG filtering and session processing
│ ├── models/ # ML and DL training scripts
│ ├── evaluation/ # Cross-validation and performance analysis
│ └── archive/ # Old or unused scripts
├── scripts/ # CSV generation scripts
├── outputs/ # Generated CSVs (not tracked in Git)
├── requirements.txt # Project dependencies
└── README.md # Project overview

markdown
Copy
Edit

## Models Implemented

- Decision Tree
- K-Nearest Neighbour (KNN)
- Gaussian Naive Bayes
- Random Forest
- Feed-Forward Neural Network (with GridSearchCV)

## Requirements

To install dependencies, run:
pip install -r requirements.txt

csharp
Copy
Edit

## Dataset

This project uses the [SEED-V EEG Dataset](https://bcmi.sjtu.edu.cn/~seed/) for training and evaluation.  
**Note:** Raw `.cnt` files are not included in this repository due to size and licensing.  
To run preprocessing, download the dataset and update paths accordingly in the scripts.

## Author

Made with  by [Muhammad Toqeer](https://github.com/MuhammadToqeer/)  
Connect on [LinkedIn](https://www.linkedin.com/in/muhammadtoqeer/)

## Status

This project is under development. More features and improvements are planned.