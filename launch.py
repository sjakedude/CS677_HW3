import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/data_banknote_authentication.txt")

# ====================
# Question #1
# ====================

# Adding Color column based on class
def color_label(row):
    if row["class"] == 0:
        return "+"
    else:
        return "-"


df["Color"] = df.apply(lambda row: color_label(row), axis=1)

# Computing mean and standard deviation for each Color (+ and -)
green_mean = round(df.loc[df["Color"] == "+"].mean(0), 2)
red_mean = round(df.loc[df["Color"] == "-"].mean(0), 2)
mean = round(df.mean(0), 2)

green_std = round(df.loc[df["Color"] == "+"].std(0), 2)
red_std = round(df.loc[df["Color"] == "-"].std(0), 2)
std = round(df.std(0), 2)

table = pd.DataFrame(
    [
        [
            red_mean["f1"],
            red_std["f1"],
            red_mean["f2"],
            red_std["f2"],
            red_mean["f3"],
            red_std["f3"],
            red_mean["f4"],
            red_std["f4"],
        ],
        [
            green_mean["f1"],
            green_std["f1"],
            green_mean["f2"],
            green_std["f2"],
            green_mean["f3"],
            green_std["f3"],
            green_mean["f4"],
            green_std["f4"],
        ],
        [
            mean["f1"],
            std["f1"],
            mean["f2"],
            std["f2"],
            mean["f3"],
            std["f3"],
            mean["f4"],
            std["f4"],
        ],
    ],
    index=[0, 1, "all"],
    columns=["µ(f1)", "σ(f1)", "µ(f2)", "σ(f2)", "µ(f3)", "σ(f3)", "µ(f4)", "σ(f4)"],
)

print(table)

print("=====")

# ====================
# Question #2
# ====================
x_train, x_test = np.split(df.sample(frac=1), 2)

x_train_green = x_train.loc[x_train["Color"] == "+"]
x_train_red = x_train.loc[x_train["Color"] == "-"]
x_test_green = x_test.loc[x_test["Color"] == "+"]
x_test_red = x_test.loc[x_test["Color"] == "-"]

seaborn.pairplot(x_train_green)
plt.savefig("output/good_bills.pdf")
plt.clf()
seaborn.pairplot(x_train_red)
plt.savefig("output/fake_bills.pdf")

# Checking each row in testing set
# 3 rules to be a good bill:
# Rule 1: f1 > -4
# Rule 2: f2 > -10
# Rule 3: -4 > f3 > -6

tp = 0
fp = 0
tn = 0
fn = 0

for index, row in x_test.iterrows():
    # We are predicting a good bill if this is true
    if row["f1"] > -4 and row["f2"] > -10 and -4 > row["f3"] > -6:
        # Check if prediction is correct
        if row["Color"] == "+":
            tp += 1
        else:
            fp += 1
    else:
        if row["Color"] == "-":
            tn += 1
        else:
            fn += 1

tpr = tp / (tp + fn)
tnr = tn / (tn + fp)

accuracy = round(((tp + tn) / len(x_test)) * 100, 2)

accuracy_table = pd.DataFrame(
    {"tp": [tp], "fp": fp, "tn": [tn], "fn": [fn], "accuracy": accuracy, "tpr": [tpr], "tnr": [tnr]}
)

print(accuracy_table)