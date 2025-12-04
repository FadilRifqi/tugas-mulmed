from utils import *
import dlib
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt


# cpu POS
def cpu_POS(signal, **kargs):
    """
    POS method on CPU using Numpy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10**-9
    X = signal
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * kargs['fps'])   # window length

    # stack e times fixed mat P
    P = np.array([[0, 1, -1], [-2, 1, 1]])
    Q = np.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = np.zeros((e, f))
    for n in np.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (np.mean(Cn, axis=2)+eps)
        M = np.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = np.multiply(M, Cn)

        # Projection (6)
        S = np.dot(Q, Cn)
        S = S[0, :, :, :]
        S = np.swapaxes(S, 0, 1)    # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = np.std(S1, axis=1) / (eps + np.std(S2, axis=1))
        alpha = np.expand_dims(alpha, axis=1)
        Hn = np.add(S1, alpha * S2)
        Hnm = Hn - np.expand_dims(np.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = np.add(H[:, m:(n + 1)], Hnm)

    return H

# Calculate FFT and find the dominant frequency
def calculate_bpm(signal, fs):
    # Perform FFT
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1/fs)  # Frequency bins
    fft_magnitude = np.abs(np.fft.rfft(signal))  # Magnitude of FFT

    # Find the dominant frequency
    dominant_freq = freqs[np.argmax(fft_magnitude)]  # Frequency with max magnitude

    # Convert to BPM
    bpm = dominant_freq * 60
    return bpm, dominant_freq

# Bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')  # Create bandpass filter
    y = filtfilt(b, a, data)  # Apply filter
    return y

# Capture Video using camera
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Path to the model file
predictor = dlib.shape_predictor(predictor_path)

# Initialize lists to store RGB values
r_values, g_values, b_values = [], [], []

# Create a live plot for RGB signal
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

# Plot for raw RGB signal
line_r, = ax1.plot([], [], 'r-', label='Red')
line_g, = ax1.plot([], [], 'g-', label='Green')
line_b, = ax1.plot([], [], 'b-', label='Blue')
ax1.set_xlim(0, 100)  # Adjust x-axis range as needed
ax1.set_ylim(0, 255)  # RGB values range from 0 to 255
ax1.legend()
ax1.set_title("Raw RGB Signal")
ax1.set_xlabel("Frame")
ax1.set_ylabel("Intensity")

# Plot for POS signal
line_pos, = ax2.plot([], [], 'm-', label='POS Signal')
ax2.set_xlim(0, 100)  # Adjust x-axis range as needed
ax2.set_ylim(-4, 4)  # Adjust y-axis range as needed
ax2.legend()
ax2.set_title("POS Signal")
ax2.set_xlabel("Frame")
ax2.set_ylabel("Amplitude")

# Show video captured
frame_count = 0
fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate of the video
while True:
    print()
    ret, frame = cap.read()
    if not ret:
        break
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)
    if faces:
        # Draw rectangles around detected faces
        for face in faces:
            # Draw rectangles around detected faces
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Detect facial landmarks
            # Detect facial landmarks
            landmarks = predictor(gray, face)

            # Get coordinates for the specified points
            point_16 = (landmarks.part(15).x, landmarks.part(15).y)
            point_36 = (landmarks.part(35).x, landmarks.part(35).y)
            point_12 = (landmarks.part(11).x, landmarks.part(11).y)

            # Define the points for the rectangle
            rectangle_points = np.array([point_16, point_36, point_12, point_16], dtype=np.int32)

            # Draw the rectangle
            cv2.polylines(frame, [rectangle_points], isClosed=True, color=(0, 255, 255), thickness=2)  # Yellow rectangle

            # Extract ROI (Region of Interest) based on the rectangle
            x_min = min(point_16[0], point_36[0], point_12[0])
            x_max = max(point_16[0], point_36[0], point_12[0])
            y_min = min(point_16[1], point_36[1], point_12[1])
            y_max = max(point_16[1], point_36[1], point_12[1])

            # Ensure ROI is within frame boundaries
            x_min, x_max = max(0, x_min), min(frame.shape[1], x_max)
            y_min, y_max = max(0, y_min), min(frame.shape[0], y_max)

            # Crop the ROI
            roi = frame[y_min:y_max, x_min:x_max]

            # Calculate average RGB values of the ROI
            if roi.size > 0:  # Ensure ROI is not empty
                avg_color = roi.mean(axis=(0, 1))  # Average over height and width
                r_values.append(avg_color[2])  # OpenCV uses BGR format
                g_values.append(avg_color[1])
                b_values.append(avg_color[0])

                # Update the raw RGB plot
                frame_count += 1
                line_r.set_data(range(len(r_values)), r_values)
                line_g.set_data(range(len(g_values)), g_values)
                line_b.set_data(range(len(b_values)), b_values)
                ax1.set_xlim(0, max(100, len(r_values)))  # Dynamically adjust x-axis

                # Apply POS algorithm if enough frames are collected
                rgb_signal = np.array([r_values, g_values, b_values])  # Shape: (3, frames)
                rgb_signal = rgb_signal[np.newaxis, :, :]

                # Apply POS
                pos_signal = cpu_POS(rgb_signal, fps=fps)
                if pos_signal.shape[1] > 21:  # Ensure length > padlen (21 for order=3)
                    filtered_pos_signal = bandpass_filter(pos_signal[0, :], lowcut=0.67, highcut=4.0, fs=30, order=3)

                    # Calculate BPM using FFT
                    bpm, dominant_freq = calculate_bpm(filtered_pos_signal, fs=30)
                    print(f"Dominant Frequency: {dominant_freq:.2f} Hz, BPM: {bpm:.2f}")
                    cv2.putText(frame, f"BPM: {bpm:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Update the POS plot with the filtered signal
                    line_pos.set_data(range(len(filtered_pos_signal)), filtered_pos_signal)
                    ax2.set_xlim(0, max(100, len(filtered_pos_signal)))  # Dynamically adjust x-axis
                else:
                    print("Signal too short for filtering. Collecting more frames...")


        fig.canvas.draw()
        fig.canvas.flush_events()


    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
