

# Sign Language Alphabet Recognition

## Project Overview
Sign Language Alphabet Recognition is a deep learning-based project that recognizes hand gestures representing alphabet letters using a webcam. It uses a Convolutional Neural Network (CNN) trained on the Sign Language MNIST dataset to predict alphabet signs in real time.

## Objectives
- Recognize sign language alphabet gestures using computer vision.
- Classify hand signs into corresponding English alphabet letters.
- Perform real-time hand detection and prediction through a webcam.
- Explore the applications of deep learning in accessibility and communication.

## Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy
- Convolutional Neural Network (CNN)

## Dataset
**Sign Language MNIST Dataset**

The dataset contains grayscale images of hand gestures representing American Sign Language alphabet letters. Letters **J and Z** are excluded because they require motion.

## Features
- Real-time webcam input
- Hand landmark detection using MediaPipe
- CNN-based alphabet classification
- Live prediction display
- Trained deep learning model

## Project Structure

```text
sign-language-recognisation/
│
├── collect_data.py
├── train_model.py
├── predict.py
├── requirements.txt
├── sign_language_model.keras
├── sign_mnist_train.csv
├── sign_mnist_test.csv
├── data.csv
├── hand_landmarker.task
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/mahatiponna/sign-language-recognisation.git
```

2. Navigate to the project folder:

```bash
cd sign-language-recognisation
```

3. Create a virtual environment:

```bash
python -m venv venv
```

4. Activate the virtual environment on Windows:

```bash
venv\Scripts\activate
```

5. Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

### Train the model

```bash
python train_model.py
```

### Run real-time recognition

```bash
python predict.py
```

Allow webcam access and show a hand sign in front of the camera.

Press **Q** to close the webcam window.

## Applications
- Assistive communication
- Sign language learning
- Human-computer interaction657
- Accessibility technology

## Future Enhancements
- Recognize complete words and sentences.
- Improve real-world gesture recognition accuracy.
- Develop a web-based interface.
- Support more sign language gestures.

## Author
**Mahati Sri Ambica**  
B.Tech Artificial Intelligence and Data Science

