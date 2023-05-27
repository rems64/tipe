import cv2
import matplotlib.pyplot as plt

from libtipe.core.maths import *
from libtipe.core.types import *
from libtipe.core.log import *

def draw_outlined_text(image, text, position, size=1, outline=4):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), outline)
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), 1)
    
def draw_features(image, features:list[Feature]):
    n = len(features)
    for i, feature in enumerate(features):
        cv2.circle(image, feature.image_coordinates.astype(int), 3, (i/n*255, 0, 255))
        draw_outlined_text(image, str(i), feature.image_coordinates.astype(int), .6)
    return image

def draw_coords(img, matrice, offset, thickness=2):
    pt1 = cast_down(matrice@np.array([offset[0], offset[1],  offset[2], 1]))

    # X
    pt2 = cast_down(matrice@np.array([offset[0] + 1, offset[1], offset[2], 1]))
    cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 0, 255), thickness)
    # Y
    pt2 = cast_down(matrice@np.array([offset[0], offset[1] + 1, offset[2], 1]))
    cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), thickness)
    # Z
    pt2 = cast_down(matrice@np.array([offset[0], offset[1],  offset[2] + 1, 1]))
    cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), thickness)


def draw_point(img, coords, color=(0, 255, 0), r=4):
    cv2.circle(img, (int(coords[0]), int(coords[1])), r, color, -1)


def draw_point_mat(img, matrice, point, color=(0, 255, 0)):
    draw_point(img, cast_down(matrice@np.append(point, 1)), color)