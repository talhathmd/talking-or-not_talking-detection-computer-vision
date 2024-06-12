# Talking or Not Talking Detection

This project aims to detect whether the person on screen is talking or not talking based on the distance between the upper and lower lips. The implementation uses OpenCV and dlib for facial landmark detection and measures the lip distance to determine the talking state.

## Features
- **Real-time Detection**: Uses the webcam to detect and display the talking state in real-time.
- **Facial Landmark Detection**: Utilizes dlib's pre-trained facial landmark detector to identify key points on the face.
- **Dynamic Thresholding**: Implements a buffer to smooth out detection and avoid immediate state changes when the lips touch briefly.

## Requirements
- Python 3.6+
- OpenCV
- dlib
- numpy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/talking-or-not_talking-detection-computer-vision.git
    cd talking-or-not_talking-detection-computer-vision
2. Install requirements:
    ```bash
    pip3 install opencv-python dlib
   
Ensure you have the face_weights.dat file in the project directory. This file contains the pre-trained weights for the dlib facial landmark detector. If not, you can download it from the dlib website.

Usage
To run the detection script, use the following command:

    python3 demo.py
    
## Files and Directories
- demo.py: Main script to run the real-time talking detection.
- face_weights.dat: Pre-trained weights for dlib's facial landmark detector.
- constants.py: Contains constants used in the script.
- README.md: This file.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your improvements.

## Inspiration
This project was inspired by the work done in [Computer-Vision-Lip-Reading-2.0](https://github.com/allenye66/Computer-Vision-Lip-Reading-2.0) by allenye66. The approach for detecting lip movement and the concept of using facial landmarks to determine talking states were adapted from this repository.

## Acknowledgments
Computer-Vision-Lip-Reading-2.0 by allenye66 for the inspiration and foundational approach to lip reading and detection.
