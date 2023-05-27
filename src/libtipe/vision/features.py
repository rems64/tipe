import cv2
from ..core.types import *
from ..core.log import *

def extract_features_from_chessboard(image, pattern_size=(7,7)):
    retval, corners = cv2.findChessboardCorners(image, pattern_size, flags=cv2.CALIB_CB_FAST_CHECK)
    if not retval:
        raise FeatureNotFound
    
    corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    n = corners.shape[0]
    features=np.ndarray((n,), dtype=Feature)

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp[:,1] = pattern_size[1]-objp[:,1]   # inverse les coordonnées selon Y afin d'obtenir une base directe (x,y,z) avec "z vers le haut"
    
    for i in range(n):
        image_coordinates = corners[i][0]
        features[i]=Feature(image_coordinates, objp[i])
    return features

def get_world_and_image_positions_from_features(features:list[Feature]):
    n = len(features)
    world = np.ndarray((n,3,), dtype=float)
    image = np.ndarray((n,2,), dtype=float)
    for i, feature in enumerate(features):
        world[i]=feature.world_coordinates
        image[i]=feature.image_coordinates
    return world, image