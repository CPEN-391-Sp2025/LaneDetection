# Lane Detection using OpenCV

This program performs lane detection on a video file using computer vision techniques. It processes each frame of the video to detect lane lines and overlays the detected lanes on the original video. The program leverages OpenCV and NumPy for efficient image processing and mathematical computations.

## Features
- **Edge Detection**: Uses the Canny edge detection algorithm to identify edges in the video frames.
- **Region of Interest Masking**: Focuses on a triangular region of interest to exclude irrelevant edges.
- **Line Detection and Averaging**: Detects lane lines using the Hough Transform and averages them into left and right lanes.
- **Overlay**: Draws the detected lanes on the original video frame for visualization.

---

## How the Algorithm Works
### 1. **Reading Video Input**
The program reads a video file frame by frame using OpenCV's `cv2.VideoCapture` method.

### 2. **Edge Detection**
Each frame is converted to grayscale to simplify processing, and Gaussian blur is applied to reduce noise. Then, the Canny edge detection algorithm is used to highlight edges in the frame that are potential lane markers.

### 3. **Region of Interest Masking**
A triangular region is defined, focusing only on the area of the image where lane lines are likely to appear (e.g., the lower half of the frame). This reduces the number of edges processed and improves performance.

### 4. **Line Detection with Hough Transform**
The masked edge-detected image is passed to the Hough Transform, which detects line segments that represent potential lane lines. Detected lines are represented as coordinate pairs: `[x1, y1, x2, y2]`.

### 5. **Line Averaging**
The program categorizes detected lines into left and right lanes based on their slopes. It averages the coordinates of each group to produce a single line representing each lane.

### 6. **Overlaying Lanes on Original Frame**
The averaged lane lines are drawn on a blank image, which is then overlaid on the original frame using OpenCV's `cv2.addWeighted` function. This produces the final frame with detected lanes highlighted.

### 7. **Displaying the Result**
The processed frame is displayed in a window in real-time. Users can quit the program by pressing the `q` key.

---

## Requirements
- Python 3.7 or later
- OpenCV (`cv2`) library
- NumPy (`numpy`) library

---

## Installation
1. Install Python from the official [Python website](https://www.python.org/).
2. Install the required libraries using pip:
   ```bash
   pip install opencv-python-headless numpy
   ```
   If you prefer the full OpenCV package (with GUI support), install:
   ```bash
   pip install opencv-python
   ```

---

## Usage
1. Save the program as `lane_detection.py`.
2. Run the script from the command line with the path to the video file:
   ```bash
   python lane_detection.py <video_path>
   ```
   Replace `<video_path>` with the path to your video file. For example:
   ```bash
   python lane_detection.py test_video.mp4
   ```

3. The program will display the processed video with detected lanes overlaid. Press `q` to exit the program.

---

## Example
```bash
python lane_detection.py sample_video.mp4
```
This command will process `sample_video.mp4` and display the video with lane detection.

---

## Troubleshooting
- **"Unable to open video file"**: Ensure the video file path is correct and accessible.
- **"ModuleNotFoundError: No module named 'cv2'"**: Ensure OpenCV is installed by running `pip install opencv-python`.
- **"Python version not compatible"**: Make sure you're using Python 3.7 or later.

---

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code.

---

## Acknowledgments
- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)

