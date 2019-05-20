import Commander
import Foundation
import Progress
import Python

let pd = Python.import("pandas")
let plt = Python.import("matplotlib.pyplot")

// wrapper function for shell commands
// must provide full path to executable
func shell(_ arguments: [String] = []) -> (String?, Int32) {
    let task = Process()
    task.executableURL = URL(fileURLWithPath: arguments[0])
    task.arguments = Array(arguments[1...])

    let pipe = Pipe()
    task.standardOutput = pipe
    task.standardError = pipe

    do {
        try task.run()
    } catch {
        // handle errors
        print("Error: \(error.localizedDescription)")
    }

    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    let output = String(data: data, encoding: .utf8)

    task.waitUntilExit()
    return (output, task.terminationStatus)
}

func which(_ cmd: String) -> String {
    return shell(["/usr/bin/which", cmd]).0!.replacingOccurrences(of: "\n", with: "")
}

struct CommandError: Error {
    let msg: String
}

func < (left: PythonObject, right: PythonObject) -> PythonObject {
    return left.__lt__(right)
}

let main = Group { group in
    group.command(
        "tf",
        Option("n-objects-values", default: "10 100 500 1000 1500 2000"),
        Option("device", default: "cpu"),
        Option("steps", default: 250),
        Flag("toy", default: false)
    ) { (nObjectsValues: String, device: String, steps: Int, toy: Bool) -> Void in

        var nObjectsValues = nObjectsValues.split(separator: " ")
        var steps = steps

        if toy {
            nObjectsValues = "9 10 11".split(separator: " ")
            steps = 10
        }

        print("nObjectsValues = \(nObjectsValues), device = \(device), steps = \(steps), toy = \(toy)")

        let commands = [
            "s4tf": [which("swift"), "run", "TF"],
            "tf_eager": [which("python"), "tf.py"],
            "tf_function": [which("python"), "tf.py", "--tf_function"],
        ]

        var data: [[String: String]] = []

        var bar = ProgressBar(count: commands.count * nObjectsValues.count)
        bar.next()

        for nObjects in nObjectsValues {
            for (name, command) in commands {
                let variableArguments = ["--n-objects", "\(nObjects)", "--steps", "\(steps)", "--device", "\(device)"]
                let arguments = command + variableArguments

                // print("################################")
                // print("## \(name)")
                // print("################################")
                // print(arguments)
                let (output, status) = shell(arguments)

                if let output = output {
                    // print(output)

                    let time = output
                        .split(separator: "\n")
                        .map(String.init)
                        .filter { $0.contains("Time = ") }
                        .map { $0.replacingOccurrences(of: "Time = ", with: "") }
                        .map { Float($0)! }
                        .first!

                    data.append([
                        "time": String(time),
                        "name": name,
                        "device": device,
                        "steps": String(steps),
                        "objects": String(nObjects),
                    ])

                    bar.next()

                } else {
                    throw CommandError(msg: "Error: status = \(status)")
                }
            }
        }

        let pydata = data.map { PythonObject($0) }

        let df = pd.DataFrame(pydata)

        df["time"] = df.time.astype(Python.float)
        df["steps"] = df.steps.astype(Python.int)
        df["objects"] = df.objects.astype(Python.int)

        let table = df.pivot_table(values: "time", index: "objects", columns: "name")

        // print(df)
        print(table)

        table.to_csv("\(device).csv") // , index: false)

        table.plot()
        plt.show()
    }
}

main.run()
