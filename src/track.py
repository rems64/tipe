import argparse
import pyperclip as pc
import matplotlib.pyplot as plt

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
    calibrations:list[CalibrationSave] = []
    paths = load_batch(args.data+"/**.data")
    for path in paths:
        calibrations.append(load_calibration(path))    
    pts = []
    dirs = []
    path = args.track
    images = []
    for i in range(3):
        calib = calibrations[i]
        print(path+f"/cam{i+1}/**.png")
        frame_paths = load_batch(path+f"/cam{i+1}/**.png")
        for j, frame_path in enumerate(frame_paths):
            if len(images)<=j: images.append([])
            if len(pts)<=j: pts.append([])
            if len(dirs)<=j: dirs.append([])
            image = cv2.imread(frame_path)
            # image_size = np.array(image.shape[:-1])
            images[j].append(image)
            marker = detect_markers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))[0]
            
            transform = np.block([[calib.rotation, calib.translation[:,np.newaxis]],
                                [np.zeros((1, 3)), np.ones((1,1))]])
            transform_inv = np.linalg.inv(transform)
            
            P_inv = np.linalg.inv(calib.intrinsic)
            rot_inv = np.linalg.inv(calib.rotation)
            pt = P_inv@np.append(marker, 1)
            pt = rot_inv@pt
            pts[j].append(-rot_inv@calib.translation)
            dirs[j].append(pt)
    
    locations = []
    for frame_count, frame in enumerate(images):
        inter0 = intersection(pts[frame_count][0], dirs[frame_count][0], pts[frame_count][1], dirs[frame_count][1])[0]
        inter1 = intersection(pts[frame_count][1], dirs[frame_count][1], pts[frame_count][2], dirs[frame_count][2])[0]
        inter2 = intersection(pts[frame_count][0], dirs[frame_count][0], pts[frame_count][2], dirs[frame_count][2])[0]
        inter = (inter0+inter1+inter2)/3
        locations.append(inter)
        
        # for i, image in enumerate(images):
        #     draw_point_mat(image, calibrations[i].projection, inter0, (255, 150, 83), 2)
        #     draw_point_mat(image, calibrations[i].projection, inter1, (255, 150, 83), 2)
        #     draw_point_mat(image, calibrations[i].projection, inter2, (255, 150, 83), 2)
        #     draw_point(image, matrices[i], model_points_list[i][indice])
            # draw_point_mat(image, calibrations[i].projection, dirs[i]+pts[i], (255, 150, 83), 5)
        #     cv2.imshow(f"Coordonnees {i}", image)
        
        # cv2.waitKey(0)
        
        # print("Intersection 1 found at\n", inter0)
        # print("Intersection 2 found at\n", inter1)
        # print("Intersection 3 found at\n", inter2)
        # print("Intersection mean is\n", inter)
    print(locations)
    t = [k for k in range(len(locations))]
    x = [location[0] for location in locations]
    y = [location[1] for location in locations]
    z = [location[2] for location in locations]
    plt.plot(t, x, color="r")
    plt.plot(t, y, color="g")
    plt.plot(t, z, color="b")
    plt.show()
    
    
except Exception as e:
    error(f"{e}\n{traceback.format_exc()}")