import pandas as pd
import numpy as np

df = pd.DataFrame({"A": [1,2,3,4], "B": ["A", "b", "C", "D"], "C": ["asd llc", "sdhey LLC illlc", "Inc", "sds 998234 u"]})
df["D"] = 0
df["E"] = 0

def categorizeOrgan(row):
    organ = row["C"]

    organ = str.lower(organ)

    organs = organ.split(" ")
    print(organs)
    for word in organs:
        if word == "inc":
            return "inc"
        elif word == "llc":
            return "llc"
        else:
            pass
    return "individual"
df["category"] = df.apply(categorizeOrgan, axis=1)
df = pd.get_dummies(df, columns=["category"])
print(df)
