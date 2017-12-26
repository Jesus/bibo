import os
import sys
import cv2

from bib_detection import detect_bibs
from number_recognition import detect_numbers

image_path = sys.argv[1]

# Bib detection
bibs = detect_bibs(image_path)
print(bibs)

# Number recognition
numbers = detect_numbers(image_path, bibs)
print(numbers)

for idx, bib in enumerate(bibs):
    number = numbers[idx]

    y0 = bib["top"] - 15
    x1 = bib["left"]
    y1 = bib["top"]
    x2 = bib["right"]
    y2 = bib["bottom"]
    image = cv2.imread(image_path)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)
    image = cv2.rectangle(image, (x1, y0), (x2, y1), (0, 0, 0), -1)
    image = cv2.putText(image, number, (x1, y1 - 2), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.5, (255, 255, 255), 1, 8)

cv2.imwrite('output.jpg', image)
