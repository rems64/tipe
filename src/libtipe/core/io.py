import glob
import pickle

from libtipe.core.log import *

def load_batch(path, skip_test=False):
    paths = glob.glob(path)
    if not skip_test and len(paths)<=0:
        error("Aucune image trouvée")
        raise FileNotFoundError
    return paths


def save_calibration(path, calib):
    try:
        with open(path, "wb") as f:
            pickle.dump(calib, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        error(f"Failed to save {path}")
        raise e

def load_calibration(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        error(f"Failed to load {path}")
        raise e