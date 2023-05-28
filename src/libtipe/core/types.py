import numpy as np
from libtipe.core.log import *

class FeatureNotFound(Exception): ...

class Feature:
    def __init__(self, image_coordinates=None, world_coordinates=None):
        self.image_coordinates = np.zeros((2,)) if None else np.asarray(image_coordinates)
        self.world_coordinates = np.zeros((3,)) if None else np.asarray(world_coordinates)
    def __str__(self) -> str:
        return str(self.image_coordinates)+"/"+str(self.world_coordinates)
    
    def __repr__(self) -> str:
        return self.__str__()

class CalibrationResult:
    def __init__(self, homography, translation, rotation) -> None:
        self.homography = homography
        self.translation = translation
        self.rotation = rotation
    
    def pretty_print(self):
        log("Homographie", logColor.blue)
        print(self.homography)
        log("Translation", logColor.blue)
        print(self.translation)
        log("Rotation", logColor.blue)
        print(self.rotation)
    
    def __str__(self) -> str:
        return "homography\n"+str(self.homography)+"\ntranslation\n"+str(self.translation)+"\nrotation\n"+str(self.rotation)
    
    def __repr__(self) -> str:
        return self.__str__()

class CalibrationSave:
    def __init__(self, intrinsic, extrinsic, projection, homography, translation, rotation):
        self.intrinsic = intrinsic
        self.extrinsic = extrinsic
        self.projection = projection
        self.homography = homography
        self.translation = translation
        self.rotation = rotation
        self.path = 0