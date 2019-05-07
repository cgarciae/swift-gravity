import numpy as np
import matplotlib.pyplot as plt
import fire
import time

G = 6.67408e-11


def gravity(objects):
    n_objects = objects.shape[0]

    masses = objects[:, 0]
    masses = np.expand_dims(masses, axis=0)
    masses = np.expand_dims(masses, axis=2)

    positions = objects[:, 1:4]
    velocities = objects[:, 4:7]

    positionsA = np.expand_dims(positions, axis=0)
    positionsA = np.tile(positionsA, [n_objects, 1, 1])

    positionsB = np.expand_dims(positions, axis=1)
    positionsB = np.tile(positionsB, [1, n_objects, 1])

    pos_diff = positionsA - positionsB

    radius = np.sqrt(np.square(pos_diff).sum(axis=2))
    np.fill_diagonal(radius, float("inf"))
    radius = np.expand_dims(radius, axis=2)

    acc = G * masses * pos_diff / np.power(radius, 3)
    acc = acc.sum(axis=1, keepdims=True).squeeze()

    return np.concatenate([np.zeros([n_objects, 1]), velocities, acc], axis=1)


def rk4(y, f, h: float = 0.01):

    k1 = h * f(y)
    k2 = h * f(y + k1 / 2)
    k3 = h * f(y + k2 / 2)
    k4 = h * f(y + k3)
    k = (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y + k


def simulation(objects, steps: int = 1000, h: float = 0.01, render_steps: int = 1):
    objects_arrays = []

    for _ in range(steps):
        objects = rk4(objects, gravity, h=h)
        objects_arrays.append(objects)

    return np.stack(objects_arrays)


def main(n_objects: int = 100, steps: int = 100, plot: bool = False):

    all_objects = np.concatenate(
        [
            np.array([[1.989e30] + [0, 0, 0] + [0, 0, 0]], dtype=np.float32),  # sun
            np.concatenate(
                [
                    np.ones([n_objects, 1]) * 5.972e24,
                    np.random.uniform(low=-149.6e9, high=149.6e9, size=[n_objects, 3]),
                    np.random.uniform(low=-29785, high=29785, size=[n_objects, 3]),
                ],
                axis=1,
            ),
        ]
    )

    print(f"Objects = {len(all_objects)}, steps = {steps}, plot = {plot}")

    t0 = time.time()
    objects_array = simulation(all_objects, steps=steps, h=80000, render_steps=10)

    print(f"Time = {time.time() - t0}")

    if plot:
        n_bodies = all_objects.count
        tail = 1000
        plt.ion()

        for i in range(objects_array.count):
            objects_slice = objects_array[max(0, i - tail) : (i + 1)]

            plt.clf()
            plt.gca().set_aspect(1)

            for b in range(n_bodies):
                xs = objects_slice[:, b, 1]
                ys = objects_slice[:, b, 2]

                plt.plot(xs, ys, c="k")
                plt.scatter([xs[-1]], [ys[-1]], c="b")

            plt.xlim(-149.6e9 * 2, 149.6e9 * 2)
            plt.ylim(-149.6e9 * 2, 149.6e9 * 2)
            plt.draw()
            plt.pause(0.0001)


if __name__ == "__main__":
    fire.Fire(main)
