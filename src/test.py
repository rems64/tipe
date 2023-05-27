import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--size", nargs=1, default=["9x6"])
args = parser.parse_args()

chessboard_size = args.size[0].split("x")
chessboard_size = (int(chessboard_size[0]), int(chessboard_size[1]))

from libtipe import *
import glob
import traceback


try:
    model_points_list = []
    image_points_list = []
    paths = glob.glob('../mesures/calibration_blender/*.png')
    # images = glob.glob('../mesures/calibration_ordi/images/*.jpg')
    if len(paths)<=0:
        error("Aucune image trouvée")
        exit()

    images = []
    for path in paths:
        image = cv2.imread(path)
        images.append(image)
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        model_points, image_points = get_world_and_image_positions_from_features(extract_features_from_chessboard(gray, chessboard_size))
        model_points_list.append(model_points)
        image_points_list.append(image_points)

    P, calibration_results = calibrate(model_points_list, image_points_list, skewless=True)
    # section("Matrice intrinsèque de la caméra")
    # print(P)
    # log_list("Homographies", [c.homography for c in calibration_results])
    # section("Homographies")
    # for i, v in enumerate(calibration_results):
    #     info(f"n°{i}")
    #     v.pretty_print()
    # section("Calcul de l'écart de calibration")
    # ecarts = np.array([[linalg.norm(cast_down(calibration_results[i].homography@model_points_list[i][j])-image_points_list[i][j], np.inf) for j in range(len(model_points_list[i]))] for i in range(len(calibration_results))])
    # print(ecarts.T)
    
    pts = []
    dirs = []
    matrices = []
    
    indice = 10     # Just for selecting some point
    
    for i, image in enumerate(images):
        # print("rotation\n", calibration_results[i].rotation)
        # print("vstack\n", np.vstack([calibration_results[i].rotation, np.array([0, 0, 0])]))
        # print("translation\n", np.append(calibration_results[i].translation, 1)[:,np.newaxis])
        extrinsic = np.hstack([calibration_results[i].rotation, calibration_results[i].translation[:,np.newaxis]])
        # print("extrinsic\n", extrinsic)
        matrice = P@extrinsic
        matrices.append(matrice)
        draw_coords(image, matrice, [-1,                 0,                    0])
        draw_coords(image, matrice, [-1,                 chessboard_size[1]+1, 0])
        draw_coords(image, matrice, [chessboard_size[0], 0,                    0])
        draw_coords(image, matrice, [chessboard_size[0], chessboard_size[1]+1, 0])
        
        P_inv = np.linalg.inv(P)
        # print("P_inv\n", P_inv)
        # extrinsic_inv = np.hstack([np.linalg.inv(calibration_results[i].rotation), calibration_results[i].translation[:,np.newaxis]])
        # print("extrinsic_inv\n", extrinsic_inv)
        pt = P_inv@np.append(image_points_list[i][indice], 1)
        # print("point\n", pt)
        pt = np.linalg.inv(calibration_results[i].rotation)@pt
        # print("point\n", pt)
        pts.append(-np.linalg.inv(calibration_results[i].rotation)@calibration_results[i].translation)
        dirs.append(pt)
        draw_point(image, image_points_list[i][indice], (0, 0, 255))
        # cv2.circle(image, (int(image_points_list[i][indice][0]), int(image_points_list[i][indice][1])), 3, (0, 0, 255), -1)
    
    print("Pts\n", pts)
    print("Dirs\n", dirs)
    inter0 = intersection(pts[0], dirs[0], pts[1], dirs[1])[0]
    inter1 = intersection(pts[1], dirs[1], pts[2], dirs[2])[0]
    inter2 = intersection(pts[0], dirs[0], pts[2], dirs[2])[0]
    # print("Inter 0-1 :", inter0)
    # print("Inter 1-2 :", inter1)
    # print("Inter 0-2 :", inter2)    
    inter = (inter0+inter1+inter2)/3
    for i, image in enumerate(images):
        dinter = cast_down(matrices[i]@np.append(inter, 1))-image_points_list[i][indice]
        print(f"Ecart {i} {dinter} {np.linalg.norm(dinter, 2)} pixels")
    
    for i, image in enumerate(images):
        draw_point_mat(image, matrices[i], inter0, (255, 150, 83))
        draw_point_mat(image, matrices[i], inter1, (255, 150, 83))
        draw_point_mat(image, matrices[i], inter2, (255, 150, 83))
    #     draw_point(image, matrices[i], model_points_list[i][indice])
        cv2.imshow(f"Coordonnees {i}", image)
    
    cv2.waitKey(0)
except Exception as e:
    error(f"{e}\n{traceback.format_exc()}")