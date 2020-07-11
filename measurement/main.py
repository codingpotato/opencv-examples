import numpy as np
import math
import cv2
from matplotlib import pyplot as plt


def gray_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def find_reference_circle(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    masked = cv2.bitwise_and(image, image, mask=mask)
    gray = gray_blur(masked)
    edges = cv2.Canny(gray, 100, 255)
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1,
                               edges.shape[0]/64, param1=200, param2=10, minRadius=10, maxRadius=30)
    output = image.copy()
    size = output.shape[1] / 2
    top = (output.shape[0] - size) / 2
    bottom = top + size
    left = size / 2
    right = left + size
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles:
            for x, y, r in circle:
                if x >= left and x <= right and y >= top and y <= bottom and hsv[x, y, 0] >= lower_blue[0] and hsv[x, y, 1] <= upper_blue[0]:
                    return (x, y, r)
    return None


def find_vertical_lines(image):
    gray = gray_blur(image)

    sigma = 0.33
    median = np.median(image)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges = cv2.Canny(gray, lower, upper)

    minLineLength = image.shape[1] / 2
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=220,
                            minLineLength=minLineLength, maxLineGap=minLineLength/5)

    v_lines = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                if angle > 85 and angle < 95 or angle > -95 and angle < -85:
                    v_lines.append((x1, y1, x2, y2))
    v_lines.sort(key=lambda x: (x[0] + x[2]) / 2)
    return v_lines


image = cv2.imread("measurement/tree.jpg", cv2.IMREAD_COLOR)

x, y, r = find_reference_circle(image)
lines = find_vertical_lines(image)

result = image.copy()

cv2.circle(result, (x, y), r, (0, 255, 0), 2)
cv2.circle(result, (x, y), 2, (0, 255, 0), 2)
x1, y1, x2, y2 = lines[0]
cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
x1, y1, x2, y2 = lines[-1]
cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Origin", image)
cv2.imshow("Result", result)
cv2.waitKey()
cv2.destroyAllWindows()
