#!/usr/bin/env python3
"""
Camera IRP for Jetson
Simple vision sensor using OpenCV.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional

class CameraIRP:
    """
    IRP for camera input on Jetson.

    Provides:
    - Motion detection
    - Face detection (using Haar cascades)
    - Simple object detection
    """

    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.prev_frame = None

        # Load Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def init_state(self) -> Dict[str, Any]:
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        return {
            'initialized': True,
            'camera_id': self.camera_id,
            'resolution': (self.width, self.height)
        }

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Capture frame and detect events"""
        if self.cap is None or not self.cap.isOpened():
            return {
                'success': False,
                'error': 'Camera not initialized'
            }

        ret, frame = self.cap.read()
        if not ret:
            return {
                'success': False,
                'error': 'Failed to capture frame'
            }

        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        events = []
        importance = 0.0

        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            events.append('face_detected')
            importance = max(importance, 0.8)

        # Motion detection (if we have previous frame)
        if self.prev_frame is not None:
            diff = cv2.absdiff(self.prev_frame, gray)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            motion_pixels = np.sum(thresh) / 255

            # Normalize motion (0-1)
            motion_normalized = min(motion_pixels / (self.width * self.height * 0.1), 1.0)

            if motion_normalized > 0.1:
                events.append('motion_detected')
                importance = max(importance, min(0.3 + motion_normalized * 0.4, 0.7))

        self.prev_frame = gray.copy()

        return {
            'success': True,
            'modality': 'vision',
            'events': events,
            'num_faces': len(faces),
            'importance': importance,
            'frame_shape': frame.shape
        }

    def energy(self, state: Dict[str, Any]) -> float:
        """Camera always has data (low energy)"""
        return 0.1

    def halt(self) -> bool:
        """Never halt (continuous monitoring)"""
        return False

    def extract(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract results"""
        return state

    def __del__(self):
        """Cleanup"""
        if self.cap is not None:
            self.cap.release()
