#PBF核心算法

import numpy as np
from config import *

def init_particles():
    positions = []
    velocities = []

    spacing = PARTICLE_RADIUS * 2.2
    start = np.array([0.5, 1.0, 0.5])

    for i in range(NUM_X):
        for j in range(NUM_Y):
            for k in range(NUM_Z):
                pos = start + np.array([i, j, k]) * spacing
                positions.append(pos)
                velocities.append(np.zeros(3))
    grid = build_neighbor_map(positions, H)
    densities = []

    for i in range(len(positions)):
        neighbors = find_neighbors(i, positions, grid, H)
        rho_i = compute_density(i, positions, neighbors, H, PARTICLE_MASS)
        densities.append(rho_i)

    print("Average initial density:", np.mean(densities))

    return np.array(positions), np.array(velocities)


def predict_positions(positions, velocities):
    """
    根据当前速度和重力，预测下一时刻粒子的位置
    """
    #1.速度更新
    velocities += GRAVITY * DT
    # velocities = np.clip(velocities, -3.0, 3.0)
    #2.位置预测
    predict_positions = positions + velocities * DT

    return predict_positions, velocities

def enforce_boundary(predicted_positions, velocities):
    """
    简单的轴对齐盒子边界约束
    """
    for i in range(len(predicted_positions)):
        for d in range(3):
            if predicted_positions[i][d] < BOUND_MIN[d]:
                predicted_positions[i][d] = BOUND_MIN[d]
                velocities[i][d] *= -0.3 #简单反弹衰减

            if predicted_positions[i][d] > BOUND_MAX[d]:
                predicted_positions[i][d] = BOUND_MAX[d]
                velocities[i][d] *= -0.3

    return predicted_positions, velocities

# def step_simulation(positions, velocities):
#     prev_positions = positions.copy()
#     predicted_positions, velocities = predict_positions(positions, velocities)
#     apply_constraints(predicted_positions, h=0.2, rho0=1.0, iterations=3)
#
#     # predicted_positions, velocities = enforce_boundary(predicted_positions, velocities)
#     apply_ground_constraints(predicted_positions, velocities)
#     apply_box_constraint(predicted_positions, velocities)
#     velocities = (predicted_positions - prev_positions) / DT
#     apply_xsph(predicted_positions, velocities)
#     return predicted_positions, velocities
def step_simulation(positions, velocities):
    prev_positions = positions.copy()

    velocities += GRAVITY * DT
    predicted_positions = positions + velocities * DT

    grid = build_neighbor_map(predicted_positions, H)
    neighbors = [len(find_neighbors(i, predicted_positions, grid, H))
                 for i in range(len(predicted_positions))]
    print("avg neighbors before constraints:", sum(neighbors) / len(neighbors))

    apply_constraints(
        predicted_positions,
        h=H,
        rho0=RHO0,
        mass=PARTICLE_MASS,
        iterations=ITERATIONS,
    )

    # 这里再算一次，看约束有没有把大家推飞
    grid = build_neighbor_map(predicted_positions, H)
    neighbors = [len(find_neighbors(i, predicted_positions, grid, H))
                 for i in range(len(predicted_positions))]
    print("avg neighbors after constraints:", sum(neighbors) / len(neighbors))

    apply_box_constraints(predicted_positions, velocities, BOUND_MIN, BOUND_MAX)
    velocities[:] = (predicted_positions - prev_positions) / DT
    # 粘性（小一点）
    apply_viscosity(predicted_positions, velocities, h=H, nu=0.03)
    # XSPH 可以开很小或者先关掉
    # apply_xsph(predicted_positions, velocities, h=H, c=XSPH_C)
    # apply_vorticity_confinement(predicted_positions, velocities, h=H, epsilon=0.1)
    apply_surface_tension(predicted_positions, velocities, h=H, k_surface=0.05, threshold=0.15)
    return predicted_positions, velocities

# 平滑核函数，用于将离散粒子转化为连续密度场
def poly6(r, h):
    if 0 <= r <= h:
        return (315 / (64 * np.pi * h**9)) * (h ** 2 - r ** 2)**3
    return 0.0

#核函数梯度，用于计算密度对位置的梯度
def spiky_grad(r_vec, h):
    r = np.linalg.norm(r_vec)
    if 0 < r <= h:
        return -45 / (np.pi * h**6) * (h - r)**2 * (r_vec / r) ###
    return np.zeros(3)

#计算粒子密度
def compute_density(i, positions, neighbors, h, mass):
    rho = mass * poly6(0.0, h)  # self term
    for j in neighbors:
        r = np.linalg.norm(positions[i] - positions[j])
        rho += mass * poly6(r, h)
    return rho

#计算拉格朗日乘子lambda
def compute_lambda(i, positions, neighbors, h, rho0, mass, eps=1e-6):
    rho_i = compute_density(i, positions, neighbors, h, mass)
    Ci = rho_i / rho0 - 1.0

    grad_i = np.zeros(3)
    sum_grad_sq = 0.0

    for j in neighbors:
        grad_ij = spiky_grad(positions[i] - positions[j], h)
        grad_i += grad_ij
        sum_grad_sq += np.dot(grad_ij, grad_ij)

    # self gradient
    sum_grad_sq += np.dot(grad_i, grad_i)

    return -Ci / (sum_grad_sq + eps)

#计算位置修正   ###
def compute_position_correction(i, positions, neighbors, lambdas, h, mass):
    dp = np.zeros(3)
    for j in neighbors:
        r_vec = positions[i] - positions[j]
        grad = spiky_grad(r_vec, h)
        scorr = artificial_pressure(np.linalg.norm(r_vec), h)
        dp += (lambdas[i] + lambdas[j]) * grad   # 注意这里的负号
    return dp

#应用约束
def apply_constraints(positions, h, rho0, mass, iterations=5):
    N = len(positions)

    for _ in range(iterations):
        grid = build_neighbor_map(positions, h)
        neighbors_list = [find_neighbors(i, positions, grid, h) for i in range(N)]

        lambdas = np.zeros(N)
        for i in range(N):
            lambdas[i] = compute_lambda(i, positions, neighbors_list[i], h, rho0, mass)

        for i in range(N):
            dp = compute_position_correction(i, positions, neighbors_list[i], lambdas, h, mass)
            positions[i] += dp
        # print("avg lambda:", np.mean(lambdas))
        # dp_list = []
        # for i in range(N):
        #     dp = compute_position_correction(i, positions, neighbors_list[i], lambdas, h, rho0, mass)
        #     dp_list.append(np.linalg.norm(dp))
        # print("avg dp:", np.mean(dp_list))

def apply_ground_constraints(positions, velocities, ground_y=0.0, dampling=0.5):
    for i in range(len(positions)):
        if positions[i][1] < ground_y:
            positions[i][1] = ground_y
            if velocities[i][1] < 0:
                velocities[i][1] *= -dampling

def apply_box_constraints(positions, velocities, bound_min, bound_max, damping=0.5):
    for i in range(len(positions)):
        for axis in range(3):
            if positions[i][axis] < bound_min[axis]:
                penetration = bound_min[axis] - positions[i][axis]
                positions[i][axis] = bound_min[axis] + 0.01 * penetration
                velocities[i][axis] *= -damping

            elif positions[i][axis] > bound_max[axis]:
                penetration = positions[i][axis] - bound_max[axis]
                positions[i][axis] = bound_max[axis] - 0.01 * penetration
                velocities[i][axis] *= -damping


def apply_xsph(positions, velocities, h, c):
    N = len(positions)
    grid = build_neighbor_map(positions, h)
    new_vel = velocities.copy()

    for i in range(N):
        neighbors = find_neighbors(i, positions, grid, h)
        vsum = np.zeros(3)
        wsum = 0.0
        for j in neighbors:
            r = np.linalg.norm(positions[i] - positions[j])
            w = poly6(r, h)  # can also use spiky kernel magnitude, poly6 is fine for smoothing
            vsum += w * (velocities[j] - velocities[i])
            wsum += w
        if wsum > 0.0:
            new_vel[i] += c * vsum
    velocities[:] = new_vel

def artificial_pressure(r, h, k=0.0001, n=4):
    dq = 0.05 * h
    if r >= dq:
        return 0.0
    w_r = poly6(r, h)
    w_dq = poly6(dq, h)
    if w_dq <= 0.0:
        return 0.0
    s = -k * (w_r / w_dq) ** n
    # clamp to avoid extreme pushes
    return max(s, -1e-3)

    # return 0.0

def _cell_index(pos, h):
    cell_size = h
    return tuple((pos // cell_size).astype(int))

def build_neighbor_map(positions, h):
    grid = {}
    for idx, p in enumerate(positions):
        cell = _cell_index(p, h)
        grid.setdefault(cell, []).append(idx)
    return grid

def find_neighbors(i, positions, grid, h):
    p = positions[i]
    base = _cell_index(p, h)
    neighbors = []
    # search 27 neighboring cells
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cell = (base[0]+dx, base[1]+dy, base[2]+dz)
                if cell in grid:
                    for j in grid[cell]:
                        if j == i:
                            continue
                        r = np.linalg.norm(positions[i] - positions[j])
                        if r <= h:
                            neighbors.append(j)
    return neighbors

def apply_vorticity_confinement(positions, velocities, h, epsilon=0.3):
    """
    涡度增强（vorticity confinement）
    positions: (N, 3)
    velocities: (N, 3) in-place update
    h: 核半径
    epsilon: 涡度增强系数（0.1~0.5 之间调）
    """
    N = len(positions)
    grid = build_neighbor_map(positions, h)

    # 1. 计算每个粒子的涡度 ω_i
    omegas = np.zeros_like(velocities)  # shape (N, 3)

    for i in range(N):
        neighbors = find_neighbors(i, positions, grid, h)
        omega = np.zeros(3)
        for j in neighbors:
            r_ij = positions[i] - positions[j]
            grad_w = spiky_grad(r_ij, h)
            omega += np.cross(velocities[j] - velocities[i], grad_w)
        omegas[i] = omega

    # 2. 计算 |ω_i| 的梯度并施加涡度力
    v_new = velocities.copy()

    for i in range(N):
        neighbors = find_neighbors(i, positions, grid, h)
        omega_i = omegas[i]
        mag_omega_i = np.linalg.norm(omega_i)
        if mag_omega_i == 0.0:
            continue

        # ∇|ω|
        grad_mag = np.zeros(3)
        for j in neighbors:
            r_ij = positions[i] - positions[j]
            grad_w = spiky_grad(r_ij, h)
            grad_mag += grad_w * np.linalg.norm(omegas[j])

        N_vec = grad_mag
        N_len = np.linalg.norm(N_vec)
        if N_len > 0.0:
            N_hat = N_vec / N_len
            # f_vorticity = ε (N_hat × ω)
            f_vort = epsilon * np.cross(N_hat, omega_i)
            # 直接作为速度增量（简单处理）
            v_new[i] += f_vort * DT  # 乘以时间步长，等价于显式积分

    velocities[:] = v_new

def apply_viscosity(positions, velocities, h, nu=0.1):
    """
    粘性项（类似 SPH 的 viscosity）
    nu: 粘性系数（0.01~0.2），值越大越“稠”
    """
    N = len(positions)
    grid = build_neighbor_map(positions, h)
    v_new = velocities.copy()

    for i in range(N):
        neighbors = find_neighbors(i, positions, grid, h)
        laplace_v = np.zeros(3)
        wsum = 0.0

        for j in neighbors:
            r = np.linalg.norm(positions[i] - positions[j])
            if r <= 0.0 or r > h:
                continue
            # 简化：用 poly6 做权重
            w = poly6(r, h)
            laplace_v += w * (velocities[j] - velocities[i])
            wsum += w

        if wsum > 0.0:
            laplace_v /= wsum
            # 显式积分
            v_new[i] += nu * laplace_v * DT

    velocities[:] = v_new

def apply_surface_tension(positions, velocities, h, k_surface=0.5, threshold=0.1):
    """
    表面张力（简单版本）
    k_surface: 表面张力系数
    threshold: 认为是表面的阈值（|n_i| > threshold）
    """
    N = len(positions)
    grid = build_neighbor_map(positions, h)

    # 颜色场梯度 n_i
    normals = np.zeros_like(positions)

    for i in range(N):
        neighbors = find_neighbors(i, positions, grid, h)
        n_i = np.zeros(3)
        for j in neighbors:
            r_ij = positions[i] - positions[j]
            grad_w = spiky_grad(r_ij, h)
            n_i += grad_w
        normals[i] = n_i

    # 施加表面张力
    for i in range(N):
        n_i = normals[i]
        n_len = np.linalg.norm(n_i)
        if n_len > threshold:
            n_hat = n_i / n_len
            # f_surface = -k_surface * n_hat
            f_surface = -k_surface * n_hat
            velocities[i] += f_surface * DT

def apply_cohesion_adhesion(positions, velocities, h, cohesion_k=0.02, adhesion_k=0.02):
    """
    cohesion: 粒子间聚合力
    adhesion: 靠近边界时的吸附力（简单朝边界法线方向）
    """
    N = len(positions)
    grid = build_neighbor_map(positions, h)

    # cohesion：粒子间微弱吸引
    for i in range(N):
        neighbors = find_neighbors(i, positions, grid, h)
        force = np.zeros(3)
        for j in neighbors:
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            if r <= 0.0 or r > h:
                continue
            dir_ij = r_vec / r
            # 距离越近力越小，越远（在 h 内）力越大一点
            weight = (r / h)
            force += weight * dir_ij
        velocities[i] += cohesion_k * force * DT

    # adhesion：靠近边界的粒子往边界“吸”一点（极简）
    for i in range(N):
        p = positions[i]
        # 对每个轴，靠近 BOUND_MIN / BOUND_MAX 时增加一点吸附
        for d in range(3):
            dist_min = p[d] - BOUND_MIN[d]
            dist_max = BOUND_MAX[d] - p[d]
            if dist_min < h:
                # 靠近最小边界，往边界方向吸
                velocities[i][d] -= adhesion_k * (h - dist_min) * DT
            if dist_max < h:
                # 靠近最大边界
                velocities[i][d] += adhesion_k * (h - dist_max) * DT

def sdf_box(point, center, half_extents):
    """
    轴对齐盒子的 SDF
    center: 盒子中心
    half_extents: 半尺寸 (hx, hy, hz)
    返回: 距离 > 0 在外面，< 0 在里面
    """
    p = point - center
    q = np.abs(p) - half_extents
    outside = np.maximum(q, 0.0)
    inside = np.minimum(np.maximum(q[0], np.maximum(q[1], q[2])), 0.0)
    return np.linalg.norm(outside) + inside

def apply_sdf_boundary(positions, velocities, center, half_extents, stiffness=0.1, damping=0.5):
    """
    用 SDF 约束粒子在盒子内部（或外部）
    这里假设粒子应该在盒子内部，如果距离为正说明超出边界，就把它往里投影
    """
    for i in range(len(positions)):
        p = positions[i]
        d = sdf_box(p, center, half_extents)
        if d > 0.0:
            # 法线近似为 SDF 的梯度，这里用数值梯度
            eps = 1e-3
            grad = np.zeros(3)
            for k in range(3):
                e = np.zeros(3)
                e[k] = eps
                grad[k] = (sdf_box(p + e, center, half_extents) -
                           sdf_box(p - e, center, half_extents)) / (2 * eps)
            norm_grad = np.linalg.norm(grad)
            if norm_grad > 0.0:
                n = grad / norm_grad
                # 把粒子往内推 d 的距离
                positions[i] -= stiffness * d * n
                # 速度沿法线反弹衰减
                v_n = np.dot(velocities[i], n)
                if v_n > 0:
                    velocities[i] -= (1 + damping) * v_n * n

def compute_adaptive_dt(velocities, h, cfl_number=0.4, dt_min=0.001, dt_max=0.02):
    """
    根据 CFL 条件调整 dt: v_max * dt / h < cfl_number
    """
    v_mag = np.linalg.norm(velocities, axis=1)
    vmax = np.max(v_mag) if len(v_mag) > 0 else 0.0
    if vmax < 1e-5:
        return dt_max
    dt = cfl_number * h / vmax
    dt = max(dt_min, min(dt, dt_max))
    return dt
