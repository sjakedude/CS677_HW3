"""
Jake Stephens
Class: CS 677 - Spring 2
Date: 7/27/2021
Homework #3
Comparing knn classifier to linear regression on bank note data.
"""

import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("data/data_banknote_authentication.txt")

# ====================
# Question #1
# ====================

print("============")
print("Question 1")
print("============")

# Adding Color column based on class
def color_label(row):
    if row["class"] == 0:
        return "+"
    else:
        return "-"


df["Color"] = df.apply(lambda row: color_label(row), axis=1)

# Computing mean and standard deviation for each Color (+ and -)
green_mean = round(df.loc[df["Color"] == "+"].mean(0, numeric_only=True), 2)
red_mean = round(df.loc[df["Color"] == "-"].mean(0, numeric_only=True), 2)
mean = round(df.mean(0, numeric_only=True), 2)

green_std = round(df.loc[df["Color"] == "+"].std(0, numeric_only=True), 2)
red_std = round(df.loc[df["Color"] == "-"].std(0, numeric_only=True), 2)
std = round(df.std(0, numeric_only=True), 2)

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

# ====================
# Question #2
# ====================

print("============")
print("Question 2")
print("============")

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
    {
        "tp": [tp],
        "fp": fp,
        "tn": [tn],
        "fn": [fn],
        "accuracy": accuracy,
        "tpr": [tpr],
        "tnr": [tnr],
    }
)

print(accuracy_table)

# ====================
# Question #3
# ====================

print("============")
print("Question 3")
print("============")

y = df["class"].values.tolist()
x = df.drop(["Color", "class"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, random_state=1, shuffle=True
)

k = [3, 5, 7, 9, 11]
accuracy_table = []

for num in k:

    knn = make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=num, weights="distance")
    )
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = metrics.accuracy_score(y_pred, y_test)
    accuracy_table.append(accuracy)

    print("k=" + str(num) + " - Accuracy:" + str(accuracy))

max_value = max(accuracy_table)
max_index = accuracy_table.index(max_value)

plt.clf()
plt.plot(accuracy_table)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()
print(
    "Best value for K is k="
    + str(k[max_index])
    + " with a accuracy of "
    + str(accuracy_table[max_index])
)

knn = KNeighborsClassifier(n_neighbors=k[max_index])
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
index = 0
tp = 0
fp = 0
tn = 0
fn = 0

for item in y_pred:
    if item == 0:
        if item == y_test[index]:
            tp += 1
        else:
            fp += 1
    else:
        if item == y_test[index]:
            tn += 1
        else:
            fn += 1
    index += 1
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)

accuracy_table = pd.DataFrame(
    {
        "tp": [tp],
        "fp": fp,
        "tn": [tn],
        "fn": [fn],
        "accuracy": accuracy_table[max_index],
        "tpr": [tpr],
        "tnr": [tnr],
    }
)

print(accuracy_table)

# BU ID Classifier: 1161

knn = KNeighborsClassifier(n_neighbors=k[max_index])
knn.fit(x_train, y_train)
bu_id_bill = pd.DataFrame([[1, 1, 6, 1]], columns=["f1", "f2", "f3", "f4"])
y_pred = knn.predict(bu_id_bill)
print(
    "The predicted value for my BU ID Bill using Knn where k="
    + str(k[max_index])
    + " is: "
    + str(y_pred)
)
if y_pred[0] == 0:
    print("My Bill is Real!!!!")
else:
    print("My Bill is Fake...")

label = None
if (
    bu_id_bill.loc[0]["f1"] > -4
    and bu_id_bill.loc[0]["f2"] > -10
    and -4 > bu_id_bill.loc[0]["f3"] > -6
):
    label = 0
else:
    label = 1
print(
    "The predicted value for my BU ID Bill using my simple classifier is: " + str(label)
)
if label == 0:
    print("My Bill is Real!!!!")
else:
    print("My Bill is Fake...")

# ====================
# Question #4
# ====================

print("============")
print("Question 4")
print("============")

columns = ["f1", "f2", "f3", "f4"]

for f in columns:
    knn = KNeighborsClassifier(n_neighbors=k[max_index])
    knn.fit(x_train.drop([f], axis=1), y_train)
    y_pred = knn.predict(x_test.drop([f], axis=1))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy Without " + f + ": " + str(accuracy))

# ====================
# Question #5
# ====================

print("============")
print("Question 5")
print("============")

clf = LogisticRegression(random_state=0).fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)

index = 0
tp = 0
fp = 0
tn = 0
fn = 0

for item in y_pred:
    if item == 0:
        if item == y_test[index]:
            tp += 1
        else:
            fp += 1
    else:
        if item == y_test[index]:
            tn += 1
        else:
            fn += 1
    index += 1
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)

accuracy_table = pd.DataFrame(
    {
        "tp": [tp],
        "fp": fp,
        "tn": [tn],
        "fn": [fn],
        "accuracy": [accuracy],
        "tpr": [tpr],
        "tnr": [tnr],
    }
)

print(accuracy_table)

# BU ID Classifier: 1161

clf = LogisticRegression(random_state=0).fit(x_train, y_train)
y_pred = clf.predict(bu_id_bill)
print(
    "The predicted value for my BU ID Bill using Logistic Regression is: " + str(y_pred)
)
if y_pred[0] == 0:
    print("My Bill is Real!!!!")
else:
    print("My Bill is Fake...")

# ====================
# Question #6
# ====================

print("============")
print("Question 6")
print("============")

columns = ["f1", "f2", "f3", "f4"]

for f in columns:
    clf = LogisticRegression(random_state=0).fit(x_train.drop([f], axis=1), y_train)
    y_pred = knn.predict(x_test.drop([f], axis=1))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy Without " + f + ": " + str(accuracy))
