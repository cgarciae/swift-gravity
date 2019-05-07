import Commander // @kylef
import Foundation
import Python

let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")

let G: PythonObject = 6.67408e-11
let DOTS = Python.slice(Python.None, Python.None, Python.None)
let ELL = Python.Ellipsis

func from(_ a: Int) -> PythonObject {
    return slice(from: PythonObject(a))
}

func to(_ a: Int) -> PythonObject {
    return slice(to: PythonObject(a))
}

func slice(_ from: PythonObject = Python.None, _ to: PythonObject = Python.None, _ stride: PythonObject = Python.None) -> PythonObject {
    return slice(from: from, to: to, stride: stride)
}

func slice(from: PythonObject = Python.None, to: PythonObject = Python.None, stride: PythonObject = Python.None) -> PythonObject {
    return Python.slice(from, to, stride)
}

func gravity(_ objects: PythonObject) -> PythonObject {
    let nObjects = objects.shape[0]

    var masses = objects[DOTS, 0]
    masses = np.expand_dims(masses, axis: 0)
    masses = np.expand_dims(masses, axis: 2)

    let positions = objects[DOTS, slice(1, 4)]
    let velocities = objects[DOTS, slice(4, 7)]

    var positionsA = np.expand_dims(positions, axis: 0)
    positionsA = np.tile(positionsA, [nObjects, 1, 1])

    var positionsB = np.expand_dims(positions, axis: 1)
    positionsB = np.tile(positionsB, [1, nObjects, 1])

    let pos_diff = positionsA - positionsB

    var radius = np.sqrt(np.square(pos_diff).sum(axis: 2))
    np.fill_diagonal(radius, Float.infinity)
    radius = np.expand_dims(radius, axis: 2)

    var acc = G * masses * pos_diff / np.power(radius, 3)
    acc = acc.sum(axis: 1, keepdims: true).squeeze()

    return np.concatenate([
        np.zeros([nObjects, 1]),
        velocities,
        acc,
    ], axis: 1)
}

func rk4(_ y: PythonObject, _ f: (PythonObject) -> PythonObject, h: Float = 0.01) -> PythonObject {
    let h = PythonObject(h)

    let k1 = h * f(y)
    let k2 = h * f(y + k1 / 2)
    let k3 = h * f(y + k2 / 2)
    let k4 = h * f(y + k3)
    let k = (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return y + k
}

func simulation(_ objects: PythonObject, steps: Int = 1000, h: Float = 0.01, render_steps _: Int = 1) -> PythonObject {
    var objects = objects
    var objects_arrays: [PythonObject] = []

    for _ in 0 ..< steps {
        objects = rk4(objects, gravity, h: h)

        objects_arrays.append(objects)
    }

    return np.stack(objects_arrays)
}

extension Collection where Element == String {
    func getOption(_ option: String) -> String? {
        return filter { $0.contains("--\(option)") }
            .map { $0.replacingOccurrences(of: "--\(option)=", with: "") }
            .first
    }
}

let main = command(
    Option("n-objects", default: 100),
    Option("steps", default: 100),
    Flag("plot")
) { (nObjects: Int, steps: Int, plot: Bool) in

    // let nObjects = CommandLine.arguments.getOption("n-objects").flatMap(Int.init) ?? 100
    // let steps = CommandLine.arguments.getOption("steps").flatMap(Int.init) ?? 1000
    // let plot = CommandLine.arguments.getOption("plot") != nil

    var all_objects = np.concatenate([
        np.array([[1.989e30] + [0, 0, 0] + [0, 0, 0]], dtype: np.float32), // sun
        np.concatenate([
            np.ones([nObjects, 1]) * PythonObject(5.972e24),
            np.random.uniform(low: -149.6e9, high: 149.6e9, size: [nObjects, 3]),
            np.random.uniform(low: -29785, high: 29785, size: [nObjects, 3]),
        ], axis: 1),
    ])

    print("Objects = \(all_objects.count), steps = \(steps), plot = \(plot)")

    let t0 = Date()
    let objects_array = simulation(all_objects, steps: steps, h: 80000, render_steps: 10)

    print("Time = \(Date().timeIntervalSince(t0))")

    if plot {
        let n_bodies = all_objects.count
        let tail = 1000
        plt.ion()

        for i in 0 ..< objects_array.count {
            let objects_slice = objects_array[slice(PythonObject(max(0, i - tail)), PythonObject(i + 1))]

            plt.clf()
            plt.gca().set_aspect(1)

            for b in 0 ..< n_bodies {
                let xs = objects_slice[DOTS, b, 1]
                let ys = objects_slice[DOTS, b, 2]

                plt.plot(xs, ys, c: "k")
                plt.scatter([xs[-1]], [ys[-1]], c: "b")
            }

            plt.xlim(-149.6e9 * 2, 149.6e9 * 2)
            plt.ylim(-149.6e9 * 2, 149.6e9 * 2)
            plt.draw()
            plt.pause(0.0001)
        }
    }
}

main.run()
