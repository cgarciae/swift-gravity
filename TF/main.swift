import Commander // @kylef
import Foundation
import Python
import TensorFlow

let np = Python.import("numpy")
let plt = Python.import("matplotlib.pyplot")

let G: Float = 6.67408e-11

// Numpy index shortcuts
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

// TensorFlow index shortcuts
let ellipsis = TensorRange.ellipsis
let newAxis = TensorRange.newAxis
let squeezeAxis = TensorRange.squeezeAxis

func at(_ n: Int) -> TensorRange {
    return .index(n)
}

func all(_ stride: Int = 1) -> TensorRange {
    return .partialRangeFrom(0..., stride: stride)
}

func from(_ n: Int, _ stride: Int = 1) -> TensorRange {
    return .partialRangeFrom(n..., stride: stride)
}

func upTo(_ n: Int, _ stride: Int = 1) -> TensorRange {
    return .partialRangeUpTo(..<n, stride: stride)
}

func range(_ from: Int, _ to: Int, _ stride: Int = 1) -> TensorRange {
    return .range(from ..< to, stride: stride)
}

extension Tensor {
    func tiled(_ factors: [Int]) -> Tensor {
        var tensor = self

        for (i, factor) in factors.enumerated() {
            if factor > 1 {
                tensor = Tensor(
                    concatenating: Array(repeating: tensor, count: factor),
                    alongAxis: i
                )
            }
        }

        return tensor
    }

    func tiled(_ factors: Int...) -> Tensor {
        return tiled(factors)
    }
}

func get_device(_ value: String) -> DeviceKind? {
    switch value {
    case "cpu":
        return .cpu
    case "gpu":
        return .gpu
    case "tpu":
        return .tpu
    default:
        return nil
    }
}

//////////////////////////
// begin simulation code
//////////////////////////
let main = command(
    Option("n-objects", default: 100),
    Option("steps", default: 100),
    Flag("plot"),
    Option("device", default: "gpu"),
    Flag("no-sun"),
    Flag("no-lines")
) { (nObjects: Int, steps: Int, plot: Bool, device: String, noSun: Bool, noLines: Bool) in

    let device = get_device(device)!

    var _eye: PythonObject

    if noSun {
        _eye = np.eye(nObjects, dtype: np.float32)
    } else {
        _eye = np.eye(nObjects + 1, dtype: np.float32)
    }
    np.fill_diagonal(_eye, Float.infinity)
    let eye = Tensor<Float>(numpy: _eye)!

    func gravity(_ objects: Tensor<Float>) -> Tensor<Float> {
        let nObjects = objects.shape[0]

        var masses = objects[0..., 0]
        masses = masses.expandingShape(at: 0)
        masses = masses.expandingShape(at: 2)

        let positions = objects[0..., range(1, 4)]
        let velocities = objects[0..., range(4, 7)]

        let positionsA = positions
            .expandingShape(at: 0)
            .tiled([nObjects, 1, 1])

        let positionsB = positions
            .expandingShape(at: 1)
            .tiled([1, nObjects, 1])

        let pos_diff = positionsA - positionsB

        var radius = sqrt(pos_diff.squared().sum(squeezingAxes: 2))

        radius += eye
        radius = radius.expandingShape(at: 2)

        var acc = G * masses * pos_diff / pow(radius, 3)
        acc = acc.sum(squeezingAxes: 1)

        return Tensor<Float>(concatenating: [
            Tensor(zeros: [nObjects, 1]),
            velocities,
            acc,
        ], alongAxis: 1)
    }

    func rk4(_ y: Tensor<Float>, _ f: (Tensor<Float>) -> Tensor<Float>, h: Float = 0.01) -> Tensor<Float> {
        let k1 = h * f(y)
        let k2 = h * f(y + k1 / 2)
        let k3 = h * f(y + k2 / 2)
        let k4 = h * f(y + k3)

        var k = k1 + 2 * k2
        k += 2 * k3 + k4
        k /= 6

        return y + k
    }

    func simulation(_ objects: Tensor<Float>, steps: Int = 1000, h: Float = 0.01, render_steps _: Int = 1) -> [Tensor<Float>] {
        var objects = objects
        var objects_arrays: [Tensor<Float>] = []

        for _ in 0 ..< steps {
            objects = rk4(objects, gravity, h: h)

            withDevice(.cpu) {
                objects_arrays.append(objects.expandingShape(at: 0))
            }
        }

        return objects_arrays
    }

    //////////////////////
    // CLI CODE
    ///////////////////////

    withDevice(device) {
        var all_objects = Tensor<Float>(numpy: np.concatenate([
            np.array([[1.989e30] + [0, 0, 0] + [0, 0, 0]]), // sun
            np.concatenate([
                np.ones([nObjects, 1]) * PythonObject(5.972e28), // 5.972e24),
                np.random.uniform(low: -149.6e9, high: 149.6e9, size: [nObjects, 3]),
                np.random.uniform(low: -29785, high: 29785, size: [nObjects, 3]),
            ], axis: 1),
        ]).astype(np.float32))!

        if noSun {
            all_objects = all_objects[1...]
        }

        print("Objects = \(all_objects.shape[0]), steps = \(steps), plot = \(plot), device = \(device)")

        let t0 = Date()
        let objects_array = Tensor(
            concatenating: simulation(all_objects, steps: steps, h: 80000, render_steps: 10),
            alongAxis: 0
        ).makeNumpyArray()

        print("Time = \(Date().timeIntervalSince(t0))")

        if plot {
            let n_bodies = all_objects.shape[0]
            let tail = 1000
            plt.ion()

            for i in 0 ..< Int(objects_array.shape[0])! {
                let objects_slice = objects_array[max(0, i - tail) ..< i + 1]

                plt.clf()
                plt.gca().set_aspect(1)

                for b in 0 ..< n_bodies {
                    let xs = objects_slice[0..., b, 1]
                    let ys = objects_slice[0..., b, 2]

                    if !noLines {
                        plt.plot(xs, ys, c: "k")
                    }
                    plt.scatter(xs[(-1)...], ys[(-1)...], c: "b")
                }

                plt.xlim(-149.6e9 * 2, 149.6e9 * 2)
                plt.ylim(-149.6e9 * 2, 149.6e9 * 2)
                plt.draw()
                plt.pause(0.01)
            }
        }
    }
}

main.run()
