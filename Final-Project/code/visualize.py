#可视化

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_particles(positions, title="Particles"):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=45, azim=-45)

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=10
    )

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 2)

    ax.set_title(title)
    plt.show()