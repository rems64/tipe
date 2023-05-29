import numpy as np
from numpy import linalg
import scipy.optimize
import scipy.sparse.linalg
import math

from libtipe.core.log import *
from libtipe.core.types import CalibrationResult
# from ..core import *

def calibrate(model_points:list[list[np.ndarray]], image_points:list[list[np.ndarray]], skewless=False) -> tuple[np.ndarray, list[CalibrationResult]]:
    assert len(model_points)==len(image_points)
    n = len(model_points)
    additionnal_constraints = 0
    if skewless: additionnal_constraints+=1
    V = np.zeros((2*n+additionnal_constraints, 6))
    homographies = []
    for plane in range(n):
        H = estimate_homography(model_points[plane], image_points[plane])
        homographies.append(H)
        v11 = np.array([H[0,0]*H[0,0],
                        H[0,0]*H[1,0] + H[1,0]*H[0,0],
                        H[1,0]*H[1,0],
                        H[2,0]*H[0,0] + H[0,0]*H[2,0],
                        H[2,0]*H[1,0] + H[1,0]*H[2,0],
                        H[2,0]*H[2,0]])
        
        v12 = np.array([H[0,0]*H[0,1],
                        H[0,0]*H[1,1] + H[1,0]*H[0,1],
                        H[1,0]*H[1,1],
                        H[2,0]*H[0,1] + H[0,0]*H[2,1],
                        H[2,0]*H[1,1] + H[1,0]*H[2,1],
                        H[2,0]*H[2,1]])
        
        v22 = np.array([H[0,1]*H[0,1],
                        H[0,1]*H[1,1] + H[1,1]*H[0,1],
                        H[1,1]*H[1,1],
                        H[2,1]*H[0,1] + H[0,1]*H[2,1],
                        H[2,1]*H[1,1] + H[1,1]*H[2,1],
                        H[2,1]*H[2,1]])

        V[2*plane] = v12
        V[2*plane+1] = v11-v22
    constraint = 2*n
    if skewless:
        V[constraint]=np.array([0., 1., 0., 0., 0., 0.])
        constraint+=1

    w, v = linalg.eig(np.transpose(V)@V)
    minimum = math.inf
    minimum_index = 0
    for i, m in enumerate(w):
        if m<minimum and not v[i].all()==0:
            minimum_index = i
            minimum = m
    b = v[:,minimum_index]

    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])
    
    v0 = (B[0,1]*B[0,2] - B[0,0]*B[1,2])/(B[0,0]*B[1,1]-B[0,1]**2)
    lambd = B[2,2] - (B[0,2]**2 + v0*(B[0,1]*B[0,2]-B[0,0]*B[1,2]))/B[0,0]
    alpha = np.sqrt(lambd/B[0,0])
    beta = np.sqrt(lambd*B[0,0]/(B[0,0]*B[1,1] - B[0,1]**2))
    gamma = -B[0,1]*alpha**2*beta/lambd
    u0 = gamma*v0/beta - B[0,2]*alpha**2/lambd
    
    A = np.array([[alpha*lambd, 0    , u0*lambd],
                  [0    , beta*lambd , v0*lambd],
                  [0    , 0          , 1 ]])
    
    A_inv = linalg.inv(A)
    results = []
    for image in range(n):
        H = homographies[image]
        h1 = H[:,0]
        h2 = H[:,1]
        h3 = H[:,2]
        l1 = 1/linalg.norm(A_inv@h1)
        l2 = 1/linalg.norm(A_inv@h2)
        l = (l1+l2)/2
        r1 = l*A_inv@h1
        r1/=linalg.norm(r1)
        r2 = l*A_inv@h2
        r2/=linalg.norm(r2)
        r3 = np.cross(r1, r2)
        r3/=linalg.norm(r3)
        R = np.hstack((np.atleast_2d(r1).T, np.atleast_2d(r2).T, np.atleast_2d(r3).T))
        t = l*A_inv@h3
        results.append(CalibrationResult(H, t, R))
    
    return (A, results)    

def estimate_homography(model_points, image_points):
    model_points_l = np.copy(model_points)
    n = len(model_points_l)
    L = np.ndarray((2*n,9))
    for i in range(n):
        # Homogeneous coordinates would require a vector of length 4 but here as z=0, we use [X,Y]
        # Note that Mt is already the transposed vector
        Mt = model_points_l[i]
        Mt[2]=1
        u, v = image_points[i]
        L[2*i] = np.hstack((Mt, np.zeros(3), -u*Mt))
        L[2*i+1] = np.hstack((np.zeros(3), Mt, -v*Mt))

    # A normalization would be appreciated and give better results    
    w, v = linalg.eig(np.transpose(L)@L)
    minimum = math.inf
    minimum_index = 0
    for i, m in enumerate(w):
        if m<minimum and not v[i].all()==0:
            minimum_index = i
            minimum = m
    H = np.zeros((3, 3))
    x = v[:,minimum_index]
    for i in range(3):
        for j in range(3):
            H[i][j] = x[3*i+j]
    
    res = scipy.optimize.least_squares(f, H.flatten(), args=(image_points, model_points_l))
    if not res.success:
        warn("The least squares homography estimate didn't converge")
    return res.x.reshape((3, 3))/res.x[-1] if res.success else H/H[2,2]


def f(h_f, m, M):
    n = len(m)
    h = h_f.reshape((3,3))
    h1 = np.array([h[0,0], h[0,1], h[0,2]])
    h2 = np.array([h[1,0], h[1,1], h[1,2]])
    h3 = np.array([h[2,0], h[2,1], h[2,2]])
    mt= np.zeros((n,2))
    for i in range(n):
        mt[i] = 1/(h3@M[i])*np.transpose(np.array([h1@M[i],h2@M[i]]))
    r = np.zeros(n)
    for i in range(n):
        diff = m[i]-mt[i]
        r[i] = diff[0]**2+diff[1]**2
    return 100*r