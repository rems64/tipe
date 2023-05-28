import numpy as np

def cast_down(v):
    if(v[-1]==0): return v[:-1]
    return v[:-1]/v[-1]

def intersection(p1, u1, p2, u2):
    v = np.cross(u2, u1)
    A = np.array([[u1[0], -u2[0], v[0]],
                  [u1[1], -u2[1], v[1]],
                  [u1[2], -u2[2], v[2]]])
    B = np.array([[p2[0]-p1[0]],
                  [p2[1]-p1[1]],
                  [p2[2]-p1[2]]])
    T = np.linalg.solve(A, B).flatten()
    I1 = p1+T[0]*u1
    I2 = p2+T[1]*u2
    # distance = np.linalg.norm(I2-I1, 2)
    milieu = (I1+I2)/2
    distance = np.linalg.norm(v, 2)*abs(T[2])
    return (milieu, distance)

# print("intersection\n", intersection(np.array([1, 0, 0]), np.array([-2, 2, 2]), np.array([-1, 0, 0]), np.array([1, 1, 1])))