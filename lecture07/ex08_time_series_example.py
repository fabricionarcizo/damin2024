#
# MIT License
#
# Copyright (c) 2024 Fabricio Batista Narcizo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#/

"""This script demonstrates a simple hand tracking plot using pyqtgraph and
MediaPipe Hands."""

# Import the required libraries.
from collections import deque

import cv2 as cv
import mediapipe as mp
import pyqtgraph as pg

from pyqtgraph.Qt import QtCore


# Initialize MediaPipe Hand module.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Setup the window for real-time plotting using pyqtgraph.
app = pg.mkQApp("Real-Time Hand Tracking Plot")
win = pg.GraphicsLayoutWidget(show=True, title="Hand Tracking Plot")
win.resize(1000, 600)
win.setWindowTitle("pyqtgraph: Hand Tracking")


# Initialize three curves for x, y, and z values.
p1 = win.addPlot(title="Hand Landmark Movement (Index Finger Tip)")
curve_x = p1.plot(pen=pg.mkPen(color=(255, 0, 0), width=4), name="X")
curve_y = p1.plot(pen=pg.mkPen(color=(0, 255, 0), width=4), name="Y")
curve_z = p1.plot(pen=pg.mkPen(color=(0, 0, 255), width=4), name="Z")


# Use deque for real-time data storage with a fixed max length.
data_x = deque(maxlen=50)
data_y = deque(maxlen=50)
data_z = deque(maxlen=50)


def update_plot():
    """Update the pyqtgraph plot with new data."""
    curve_x.setData(data_x)
    curve_y.setData(data_y)
    curve_z.setData(data_z)


# Setup the timer to trigger plot updates at intervals.
timer = QtCore.QTimer()
timer.timeout.connect(update_plot)
timer.start(10)


# Open the video capture.
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            # Capture frame-by-frame.
            ret, frame = cap.read()

            # Exit if frame is not captured correctly.
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Convert the frame to RGB (since OpenCV uses BGR).
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            # Process the image for hand landmarks.
            results = hands.process(image)

            # If landmarks are detected, process and plot data.
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the frame.
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Get the x, y, and z coordinates of the index finger tip.
                    index_tip = hand_landmarks.landmark[
                        mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    data_x.append(index_tip.x)
                    data_y.append(1. - index_tip.y)
                    data_z.append(index_tip.z)

            # Display the frame with hand landmarks in a window.
            cv.imshow("Hand Tracking", frame)

            # Break the loop if "q" is pressed.
            if cv.waitKey(1) == ord("q"):
                break

finally:
    # Release resources properly after finishing.
    cap.release()
    cv.destroyAllWindows()
