#全局参数

import numpy as np

#时间参数
DT = 0.01
GRAVITY = np.array([0.0, -9.8, 0.0])

#粒子参数
PARTICLE_RADIUS = 0.05
REST_DENSITY = 672.9
# REST_DENSITY = 7859.2
PARTICLE_MASS = 1.0

#流体参数
H = PARTICLE_RADIUS * 4.0
RHO0 = REST_DENSITY
ITERATIONS = 8
XSPH_C = 0.02

#场景
NUM_X = 10
NUM_Y = 10
NUM_Z = 10


#边界
BOUND_MIN = np.array([0.0, 0.0, 0.0])
BOUND_MAX = np.array([2.0, 2.0, 2.0])
