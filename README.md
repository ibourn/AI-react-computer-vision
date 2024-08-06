# React Computer Vision Project

> Playing and discovering computer vision with TensorFlow.js.

This project is a React application that utilizes TensorFlow.js for computer vision tasks, including face detection and object detection. The application accesses the user's camera to draw face meshes, pose landmarks, and bounding boxes around detected objects.

## Features

- Face Landmark Detection: Detects facial landmarks and draws meshes on the user's face.
- Pose Estimation: Uses PoseNet to estimate body pose landmarks.
- Object Detection: Utilizes COCO-SSD to detect common objects in the camera feed.

## Getting Started

**Prerequisites**  
Node.js (version 16.16.0 recommended)  
Yarn package manager

**Installation**

1. Clone the repository:

   ```
   git clone https://github.com/ibourn/AI-react-computer-vision.git
   cd AI-react-computer-vision
   ```

2. Install dependencies using npm:

   ```
   npm install --legacy-peer-deps
   ```

3. Alternatively, if you prefer using Yarn, run:
   ```
   yarn install
   ```

**Running the Application**  
To start the application, run:

```
    yarn start
```

or

```
    npm start
```

## Usage

1. Allow the application to access your camera when prompted.
2. The application will draw:

- Face meshes for detected faces.
- Pose landmarks when the user is at an appropriate distance from the camera.
- Bounding boxes around detected objects with their labels (e.g., person, bottle, phone).

## Models Used

- Face Landmark Detection: TensorFlow.js model for detecting facial landmarks.
- PoseNet: A model for estimating human poses.
- Coco-ssd: A model for detecting objects in images.
