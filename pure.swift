// import Commander
import Foundation
import Python

let plt = Python.import("matplotlib.pyplot")

let G: Float = 6.67408e-11

struct Object {
    var mass: Float = 0
    var position: [Float] = [0, 0, 0]
    var velocity: [Float] = [0, 0, 0]
}

func + (left: Object, right: Object) -> Object {
    return Object(
        mass: left.mass + right.mass,
        position: zip(left.position, right.position).map(+),
        velocity: zip(left.velocity, right.velocity).map(+)
    )
}

func * (left: Object, right: Float) -> Object {
    return Object(
        mass: left.mass * right,
        position: left.position.map { $0 * right },
        velocity: left.velocity.map { $0 * right }
    )
}

func + (left: [Object], right: [Object]) -> [Object] {
    return zip(left, right).map(+)
}

func * (left: [Object], right: Float) -> [Object] {
    return left.map { $0 * right }
}

func * (left: Float, right: [Object]) -> [Object] {
    return right * left
}

func / (left: [Object], right: Float) -> [Object] {
    return left * (1.0 / right)
}

func gravity(_ objects: [Object]) -> [Object] {
    var deltas: [Object] = Array(repeating: Object(), count: objects.count)

    for i in 0 ..< objects.count {
        deltas[i].position = objects[i].velocity

        for j in (i + 1) ..< objects.count {
            let diff_pos = zip(objects[i].position, objects[j].position).map(-)
            let radius = diff_pos.map { $0 * $0 }.reduce(0, +).squareRoot()

            deltas[i].velocity = zip(deltas[i].velocity, diff_pos).map {
                $0 - G * objects[j].mass * $1 / (radius * radius * radius)
            }
            deltas[j].velocity = zip(deltas[j].velocity, diff_pos).map {
                $0 + G * objects[i].mass * $1 / (radius * radius * radius)
            }
        }
    }

    return deltas
}

func rk4(_ y: [Object], _ f: ([Object]) -> [Object], h: Float = 0.01) -> [Object] {
    let k1 = h * f(y)
    let k2 = h * f(y + k1 / 2)
    let k3 = h * f(y + k2 / 2)
    let k4 = h * f(y + k3)
    let k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
    // let k = k1

    return y + k
}

func simulation(_ objects: [Object], steps: Int = 1000, h: Float = 0.01, render_steps _: Int = 1) -> [[Object]] {
    var objects = objects
    var objects_arrays: [[Object]] = []

    for _ in 0 ..< steps {
        objects = rk4(objects, gravity, h: h)

        objects_arrays.append(objects)
    }

    return objects_arrays
}

extension Collection where Element == String {
    func getOption(_ option: String) -> String? {
        return filter { $0.contains("--\(option)") }
            .map { $0.replacingOccurrences(of: "--\(option)=", with: "") }
            .first
    }
}

var nObjects = CommandLine.arguments.getOption("n-objects").flatMap(Int.init) ?? 100
let steps = CommandLine.arguments.getOption("steps").flatMap(Int.init) ?? 1000
var plot = CommandLine.arguments.getOption("plot") != nil

let sun = Object(mass: 1.989e30, position: [0, 0, 0], velocity: [0, 0, 0])
var all_objects = [sun]

all_objects.append(contentsOf: (1 ... nObjects).map { _ in
    Object(
        mass: 5.972e24 * Float.random(in: 0.5 ... 1.5),
        position: [
            149.6e9 * (Bool.random() ? Float.random(in: 0.5 ... 1.5) : Float.random(in: -1.5 ... -0.5)),
            149.6e9 * (Bool.random() ? Float.random(in: 0.5 ... 1.5) : Float.random(in: -1.5 ... -0.5)),
            0,
        ],
        velocity: [
            29785 * (Bool.random() ? Float.random(in: 0.5 ... 1.5) : Float.random(in: -1.5 ... -0.5)),
            29785 * (Bool.random() ? Float.random(in: 0.5 ... 1.5) : Float.random(in: -1.5 ... -0.5)),
            0,
        ]
    )
})

print("Objects = \(all_objects.count), steps = \(steps), plot = \(plot)")

let t0 = Date()
let objects_array = simulation(all_objects, steps: steps, h: 80000, render_steps: 10)

print("Time = \(Date().timeIntervalSince(t0))")

if plot {
    let n_bodies = all_objects.count
    let tail = 1000
    plt.ion()

    for i in 0 ..< objects_array.count {
        let objects_slice = objects_array[max(0, i - tail) ..< i + 1]

        plt.clf()
        plt.gca().set_aspect(1)

        for b in 0 ..< n_bodies {
            let xs = objects_slice.map { $0[b].position[0] }
            let ys = objects_slice.map { $0[b].position[1] }

            plt.plot(xs, ys, c: "k")
            plt.scatter([xs.last!], [ys.last!], c: "b")
        }

        plt.xlim(-149.6e9 * 2, 149.6e9 * 2)
        plt.ylim(-149.6e9 * 2, 149.6e9 * 2)
        plt.draw()
        plt.pause(0.0001)
    }
}
