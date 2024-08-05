import React, { useRef, useEffect } from "react";
import "./App.css";
import Webcam from "react-webcam";

import * as tf from "@tensorflow/tfjs";
import * as facemesh from "@tensorflow-models/face-landmarks-detection";
import * as posenet from "@tensorflow-models/posenet";
import * as cocossd from "@tensorflow-models/coco-ssd";

import { drawMesh, drawKeypoints, drawSkeleton, drawRect } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const run = async () => {
    const faceNet = await facemesh.load(
      facemesh.SupportedPackages.mediapipeFacemesh
    );
    console.log("Loaded facemesh.");
    const poseNet = await posenet.load({
      inputResolution: { width: 640, height: 480 },
      scale: 0.8,
    });
    console.log("Loaded posenet.");
    const cocoNet = await cocossd.load();
    console.log("Loaded cocossd.");

    setInterval(() => {
      detect(faceNet, poseNet, cocoNet);
    }, 25);
  };

  const detect = async (faceNet, poseNet, cocoNet) => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video.readyState === 4
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = webcamRef.current.video.videoWidth;
      const videoHeight = webcamRef.current.video.videoHeight;

      // Set canvas dimension
      canvasRef.current.width = videoWidth;
      canvasRef.current.height = videoHeight;

      // Make Detections
      const face = await faceNet.estimateFaces({ input: video });
      console.log("face: ", face);

      const pose = await poseNet.estimateSinglePose(video);
      console.log("pose:", pose);

      const obj = await cocoNet.detect(video);
      console.log("obj: ", obj);

      drawCanvas(face, pose, obj, canvasRef);
    }
  };

  const drawCanvas = (face, pose, obj, canvas) => {
    const ctx = canvas.current.getContext("2d");

    requestAnimationFrame(() => {
      // facemesh
      drawMesh(face, ctx);
      // pose
      drawKeypoints(pose["keypoints"], 0.6, ctx);
      drawSkeleton(pose["keypoints"], 0.7, ctx);
      // objects
      drawRect(obj, ctx);
    });
  };

  useEffect(() => {
    run();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />

        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zindex: 9,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;
