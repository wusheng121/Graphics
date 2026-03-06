#程序入口

from pbf import init_particles, step_simulation
from visualize import show_particles

def main():
    positions, velocities = init_particles()
    show_particles(positions, title="Initial Particle Destribution")

    for step in range(1, 101):
        #预测位置
        positions, velocities = step_simulation(positions, velocities)

        if step % 1 == 0:
            show_particles(positions, title=f"Step {step}")



if __name__ == "__main__":
    main()