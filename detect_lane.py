import cv2
import numpy as np
import sys

def calculate_line_endpoints(image, line):
    """
    Convert a slope-intercept representation of a line into pixel coordinates.

    Args:
        image (numpy.ndarray): The input image to determine height.
        line (tuple): A tuple (slope, intercept) representing the line equation y = mx + b.

    Returns:
        list: A list of coordinates [x1, y1, x2, y2] representing the endpoints of the line.
    """
    slope, intercept = line
    y1 = int(image.shape[0])  # bottom of the image
    y2 = int(y1 * 3 / 5)      # slightly lower than the middle
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_lines(image, lines):
    """
    Average multiple line segments into two main lanes: left and right.

    Args:
        image (numpy.ndarray): The input image used to define line positions.
        lines (list): A list of detected lines represented by their endpoints [x1, y1, x2, y2].

    Returns:
        list: A list containing two averaged lines (left and right lanes).
    """
    left_fit = []
    right_fit = []
    if lines is None:
        return None
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:  # Skip vertical lines to avoid division by zero
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:  # y is reversed in image
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    left_line = None
    right_line = None
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = calculate_line_endpoints(image, left_fit_average)
    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = calculate_line_endpoints(image, right_fit_average)
    averaged_lines = []
    if left_line is not None:
        averaged_lines.append(left_line)
    if right_line is not None:
        averaged_lines.append(right_line)
    return averaged_lines

def apply_canny_edge_detection(img):
    """
    Apply Canny edge detection to an input image.

    Args:
        img (numpy.ndarray): The input image in RGB format.

    Returns:
        numpy.ndarray: The edge-detected image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def draw_lines_on_image(img, lines):
    """
    Draw detected lines on a blank image.

    Args:
        img (numpy.ndarray): The input image to determine dimensions.
        lines (list): A list of line endpoints [x1, y1, x2, y2].

    Returns:
        numpy.ndarray: An image with the drawn lane lines.
    """
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image

def mask_region_of_interest(edges):
    """
    Mask the input image to focus on the region of interest (triangular ROI).

    Args:
        edges (numpy.ndarray): The edge-detected image.

    Returns:
        numpy.ndarray: A masked image with only the ROI visible.
    """
    height = edges.shape[0]
    width = edges.shape[1]
    triangle = np.array([[
        (width // 5, height),
        (width // 2, height // 2),
        (4 * width // 5, height),
    ]], np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image

def main():
    """
    Main function to process the video file for lane detection.

    Expects:
        A single command-line argument specifying the path to the video file.
    """
    if len(sys.argv) != 2:
        print("Usage: python lane_detection.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        sys.exit(1)

    while cap.isOpened():
        """
        Continuously process video frames to detect and display lane lines.

        Steps:
            1. Capture a frame from the video.
            2. Apply Canny edge detection.
            3. Mask the edge-detected image to focus on the ROI.
            4. Detect lines using the Hough Transform.
            5. Average the detected lines into left and right lanes.
            6. Draw the lane lines and overlay them on the original frame.
            7. Display the processed frame and exit on 'q' key press.
        """
        ret, frame = cap.read()
        if not ret:
            break

        edges = apply_canny_edge_detection(frame)
        cropped_edges = mask_region_of_interest(edges)
        lines = cv2.HoughLinesP(cropped_edges, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_lines(frame, lines)
        line_image = draw_lines_on_image(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

        cv2.imshow("Lane Detection + Overlay", combo_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()