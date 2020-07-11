import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

img = cv2.imread("measurement/tree.jpg", cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

sigma = 0.5
v = np.median(img)
lower = int(max(0, (1.0 - sigma) * v))
upper = int(min(255, (1.0 + sigma) * v))
edges = cv2.Canny(blurred, lower, upper)

minLineLength = img.shape[1] / 2
output = img.copy()
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=220,
                        minLineLength=minLineLength, maxLineGap=minLineLength/5)
for line in lines:
    for x1, y1, x2, y2 in line:
        angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
        if angle > 85 and angle < 95 or angle > -95 and angle < -85:
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 2)
plt.imshow(edges, cmap="gray")
plt.title('edges')
plt.xticks([]), plt.yticks([])
plt.subplot(1, 4, 3)
plt.imshow(output)
plt.title('output')
plt.xticks([]), plt.yticks([])
plt.show()
