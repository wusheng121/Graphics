import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from tqdm import tqdm

from pbf import init_particles, step_simulation

# 初始化粒子
positions, velocities = init_particles()

# 动画参数
TOTAL_FRAMES = 150

# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=45, azim=-45)
# ax.view_init(elev=90, azim=-90)

scat = ax.scatter(
    positions[:, 0],
    positions[:, 1],
    positions[:, 2],
    s=10
)

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_zlim(0, 2)

# 用于保存每一帧的粒子位置
frames_data = []

# -----------------------------
# 先运行模拟 + 保存每一帧数据（带进度条）
# -----------------------------
print("Running simulation...")
for _ in tqdm(range(TOTAL_FRAMES)):
    positions, velocities = step_simulation(positions, velocities)
    frames_data.append(positions.copy())

print("Simulation finished. Rendering animation...")

# -----------------------------
# 再用保存的数据生成动画
# -----------------------------
def update(frame):
    pos = frames_data[frame]
    scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    return scat,

ani = FuncAnimation(fig, update, frames=TOTAL_FRAMES, interval=50, blit=False)

# -----------------------------
# 保存动画（mp4）
# -----------------------------
ani.save("pbf_simulation_45.mp4", fps=30, dpi=150)

print("Animation saved as pbf_simulation.mp4")

# plt.show()
