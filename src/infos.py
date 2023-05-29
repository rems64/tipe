import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", nargs=1, default=["9x6"])
parser.add_argument("--data", required=True)
args = parser.parse_args()

chessboard_size = args.size[0].split("x")
chessboard_size = (int(chessboard_size[0]), int(chessboard_size[1]))

from libtipe import *
import glob
import traceback


try:
    images = []
    calibrations:list[CalibrationSave] = []
    paths = load_batch(args.data+"/**.data")
    for path in paths:
        calibrations.append(load_calibration(path))
    for i, calib in enumerate(calibrations):
        # info(calib.intrinsic)
        info("Path \n"+calib.path)
        print("Projection \n"+str(calib.projection))
        warn("Intrinsic \n"+str(calib.intrinsic))
        print("Extrinsic \n"+str(calib.extrinsic))
        print("Homography\n"+str(calib.homography))
        print("Translation\n"+str(calib.translation))
        print("Rotation\n"+str(calib.rotation))
        print()
        
except Exception as e:
    error(f"{e}\n{traceback.format_exc()}")