import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import fire
from time import time

G = 6.67408e-11


#############
# begin simulation code
#############
eye = None


def gravity(objects):
    global eye

    n_objects = objects.shape[0]

    if eye is None:
        eye = np.eye(n_objects, dtype=np.float32)
        np.fill_diagonal(eye, np.Infinity)
        eye = tf.constant(eye)

    masses = objects[:, 0]
    masses = tf.expand_dims(masses, axis=0)
    masses = tf.expand_dims(masses, axis=2)

    positions = objects[:, 1:4]
    velocities = objects[:, 4:7]

    positionsA = tf.expand_dims(positions, axis=0)
    positionsA = tf.tile(positionsA, [n_objects, 1, 1])

    positionsB = tf.expand_dims(positions, axis=1)
    positionsB = tf.tile(positionsB, [1, n_objects, 1])

    pos_diff = positionsA - positionsB

    radius = tf.sqrt(tf.reduce_sum(tf.square(pos_diff), axis=2))
    radius += eye
    radius = tf.expand_dims(radius, axis=2)

    acc = G * masses * pos_diff / pow(radius, 3)
    acc = tf.reduce_sum(acc, axis=1)

    return tf.concat([tf.zeros([n_objects, 1]), velocities, acc], axis=1)

    # return Tensor<Float>(concatenating = [
    #     Tensor(zeros = [n_objects, 1]),
    #     velocities,
    #     acc,
    # ], alongAxis = 1)


def rk4(y, f, h=0.01):
    k1 = h * f(y)
    k2 = h * f(y + k1 / 2)
    k3 = h * f(y + k2 / 2)
    k4 = h * f(y + k3)

    k = k1 + 2 * k2
    k += 2 * k3 + k4
    k /= 6

    return y + k


def simulation(objects, steps=1000, h=0.01, render_steps=1):
    objects_arrays = []

    for _ in range(steps):
        objects = rk4(objects, gravity, h=h)

        objects_arrays.append(tf.expand_dims(objects, axis=0).numpy())

    return np.concatenate(objects_arrays, axis=0)


###########
# CLI CODE
###########/


def main(
    n_objects=100, steps=100, plot=False, device="gpu", no_sun=False, no_lines=False
):

    with tf.device(device):
        all_objects = np.concatenate(
            [
                np.array([[1.989e30] + [0, 0, 0] + [0, 0, 0]]),  # sun
                np.concatenate(
                    [
                        np.ones([n_objects, 1]) * 5.972e28,  # 5.972e24),
                        np.random.uniform(
                            low=-149.6e9, high=149.6e9, size=[n_objects, 3]
                        ),
                        np.random.uniform(low=-29785, high=29785, size=[n_objects, 3]),
                    ],
                    axis=1,
                ),
            ]
        ).astype(np.float32)

        if no_sun:
            all_objects = all_objects[1:]

        print(
            f"Objects = {all_objects.shape[0]}, steps = {steps}, plot = {plot}, device = {device}"
        )

        t0 = time()
        objects_array = simulation(all_objects, steps=steps, h=80000, render_steps=10)

        print(f"Time = {time() - t0}")

        if plot:
            n_bodies = all_objects.shape[0]
            tail = 1000
            plt.ion()

            for i in range(len(objects_array)):
                # print(objects_array.shape)
                objects_slice = objects_array[max(0, i - tail) : (i + 1)]

                plt.clf()
                plt.gca().set_aspect(1)

                for b in range(n_bodies):
                    # print(objects_slice.shape)
                    xs = objects_slice[:, b, 1]
                    ys = objects_slice[:, b, 2]

                    if not no_lines:
                        plt.plot(xs, ys, c="k")

                    plt.scatter(xs[-1:], ys[-1:], c="b")

                plt.xlim(-149.6e9 * 2, 149.6e9 * 2)
                plt.ylim(-149.6e9 * 2, 149.6e9 * 2)
                plt.draw()
                plt.pause(0.01)


if __name__ == "__main__":
    fire.Fire(main)
