import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--size", nargs=1, default=["9x6"])
parser.add_argument("--output", default="calibration.json")
args = parser.parse_args()

chessboard_size = args.size[0].split("x")
chessboard_size = (int(chessboard_size[0]), int(chessboard_size[1]))

from libtipe import *
import traceback


try:
    images = []
    model_points_list = []
    image_points_list = []
    
    paths = load_batch("../mesures/calibration_blender_2/*.png")
    for path in paths:
        image = cv2.imread(path)
        images.append(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        model_points, image_points = get_world_and_image_positions_from_features(extract_features_from_chessboard(gray, chessboard_size))
        model_points_list.append(model_points)
        image_points_list.append(image_points)

    P, calibration_results = calibrate(model_points_list, image_points_list, skewless=True)
    for i in range(len(images)):
        extrinsic = np.hstack([calibration_results[i].rotation, calibration_results[i].translation[:,np.newaxis]])
        matrice = P@extrinsic
        calib_save = CalibrationSave(P, extrinsic, matrice, calibration_results[i].homography, calibration_results[i].translation, calibration_results[i].rotation)
        calib_save.path = str(pathlib.Path(paths[i]).absolute().resolve())
        parts = args.output.split(".")
        filepath = f"{parts[0]}_{i}.{parts[1]}"
        print(filepath)
        save_calibration(filepath, calib_save)

except Exception as e:
    error(f"{e}\n{traceback.format_exc()}")