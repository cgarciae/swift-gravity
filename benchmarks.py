import numpy as np
import fire
import subprocess
import sh

import pandas as pd

from tqdm import tqdm
import matplotlib.pyplot


class CLI:
    def n_objects(self, n_objects_values, device="cpu", steps=1000, toy=False):

        if toy:
            n_objects_values = [9, 10, 11]
            steps = 10

        commands = dict(
            s4tf=lambda n_objects: f"swift run TF --n-objects {n_objects} --steps {steps}",
            tf_eager=lambda n_objects: f"python tf.py --n-objects {n_objects} --steps {steps}",
            tf_function=lambda n_objects: f"python tf.py --tf_function --n-objects {n_objects} --steps {steps}",
        )

        bar = tqdm(total=len(commands) * len(n_objects_values))
        data = []

        for name, command_f in commands.items():
            for n_objects in n_objects_values:
                command = command_f(n_objects).split()
                lines = sh.Command(command[0])(command[1:]).split("\n")
                lines = filter(lambda line: line.startswith("Time = "), lines)
                lines = map(lambda line: float(line.replace("Time = ", "")), lines)
                time = list(lines)[0]

                data.append(dict(name=name, time=time, n_objects=n_objects))
                bar.update(1)

        df = pd.DataFrame(data)

        table = pd.pivot_table(
            df, values="time", index="n_objects", columns="name"
        ).plot()

        plt.show()

        print(df)
        print(table)


if __name__ == "__main__":
    fire.Fire(CLI)
