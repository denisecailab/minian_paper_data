# %% imports
import re
import pandas as pd

# %% read in file
with open("environment.yml", "r") as ymlf:
    raw = ymlf.readlines()
    deps = list(
        filter(
            bool,
            map(lambda l: re.search(r"\ +\-\ +([a-z 0-9]+)\=+([0-9 \.]+)", l), raw),
        )
    )
dep_df = pd.DataFrame(
    [(d.group(1), d.group(2)) for d in deps], columns=("Package", "Version")
)
dep_df.to_csv("dep.csv", index=False)
