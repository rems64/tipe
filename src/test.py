import argparse
import pyperclip as pc

parser = argparse.ArgumentParser()
parser.add_argument("--size", nargs=1, default=["9x6"])
parser.add_argument("--data", required=True)
parser.add_argument("--track", required=True)
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
    # for i, calib in enumerate(calibrations):
    #     # info(calib.intrinsic)
    #     image = cv2.imread(calib.path)
    #     draw_coords(image, calib.projection, [-1,                 0,                    0])
    #     draw_coords(image, calib.projection, [-1,                 chessboard_size[1]+1, 0])
    #     draw_coords(image, calib.projection, [chessboard_size[0], 0,                    0])
    #     draw_coords(image, calib.projection, [chessboard_size[0], chessboard_size[1]+1, 0])
    #     draw_coords(image, calib.projection, [chessboard_size[0], chessboard_size[1]+1, 1])
    #     cv2.imshow(f"Coordonnees {i}", image)
    
    # cv2.waitKey(0)
    
    pts = []
    dirs = []
    paths = load_batch(args.track)
    for i, path in enumerate(paths):
        calib = calibrations[i]
        image = cv2.imread(path)
        image_size = np.array(image.shape[:-1])
        images.append(image)
        marker = detect_markers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))[0]
        
        transform = np.block([[calib.rotation, calib.translation[:,np.newaxis]],
                              [np.zeros((1, 3)), np.ones((1,1))]])
        transform_inv = np.linalg.inv(transform)
        
        P_inv = np.linalg.inv(calib.intrinsic)
        rot_inv = np.linalg.inv(calib.rotation)
        # pt = P_inv@np.append(marker, calib.intrinsic[0][0])
        marker_normalized = marker/image_size
        pt = P_inv@np.append(marker, 1)
        # print("####")
        # print(np.linalg.inv(calib.rotation))
        # print(calib.rotation.T)
        # Now pt is in camera space
        pt = rot_inv@pt
        # pt = transform_inv@np.append(pt, 1)
        pts.append(-rot_inv@calib.translation)
        # pts.append(cast_down(transform_inv@np.array([[0],[0],[0],[1]])).T[0])
        dirs.append(pt)
        # draw_point(image, marker, (0, 0, 255))
        # cv2.imshow(f"image {i}", image)
    
    # cv2.waitKey(0)
    # print(pts)
    # print()
    # print(dirs)
    
    inter0 = intersection(pts[0], dirs[0], pts[1], dirs[1])[0]
    inter1 = intersection(pts[1], dirs[1], pts[2], dirs[2])[0]
    inter2 = intersection(pts[0], dirs[0], pts[2], dirs[2])[0]
    inter = (inter0+inter1+inter2)/3
    
    for i, image in enumerate(images):
        draw_point_mat(image, calibrations[i].projection, inter0, (255, 150, 83), 2)
        draw_point_mat(image, calibrations[i].projection, inter1, (255, 150, 83), 2)
        draw_point_mat(image, calibrations[i].projection, inter2, (255, 150, 83), 2)
    #     draw_point(image, matrices[i], model_points_list[i][indice])
        # draw_point_mat(image, calibrations[i].projection, dirs[i]+pts[i], (255, 150, 83), 5)
    #     cv2.imshow(f"Coordonnees {i}", image)
    
    # cv2.waitKey(0)
    
    print("Intersection 1 found at\n", inter0)
    print("Intersection 2 found at\n", inter1)
    print("Intersection 3 found at\n", inter2)
    print("Intersection mean is\n", inter)
    
    to_copy = str(inter[0])+"	"+str(inter[1])+"	"+str(inter[2])
    to_copy = to_copy.replace(".", ",")
    pc.copy(str(inter[2]))

except Exception as e:
    error(f"{e}\n{traceback.format_exc()}")