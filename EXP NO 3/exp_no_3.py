
import math
import csv
import os


def load_csv(filename):
    with open(filename, "r", newline="") as file:
        lines = csv.reader(file)
        dataset = list(lines)

    headers = dataset.pop(0)
    return dataset, headers


class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""


def subtables(data, col, delete):
    dic = {}

    coldata = [row[col] for row in data]
    attr = list(set(coldata))

    counts = [0] * len(attr)
    r = len(data)
    c = len(data[0])

    for x in range(len(attr)):
        for y in range(r):
            if data[y][col] == attr[x]:
                counts[x] += 1

    for x in range(len(attr)):
        dic[attr[x]] = [
            [0 for i in range(c)]
            for j in range(counts[x])
        ]

        pos = 0

        for y in range(r):
            if data[y][col] == attr[x]:
                row = data[y].copy()

                if delete:
                    del row[col]

                dic[attr[x]][pos] = row
                pos += 1

    return attr, dic


def entropy(S):
    attr = list(set(S))

    if len(attr) <= 1:
        return 0

    sums = 0

    for value in attr:
        p = S.count(value) / len(S)

        if p > 0:
            sums += -p * math.log(p, 2)

    return sums


def compute_gain(data, col):
    attr, dic = subtables(data, col, delete=False)

    total_size = len(data)

    total_entropy = entropy([row[-1] for row in data])

    for value in attr:
        ratio = len(dic[value]) / total_size
        ent = entropy([row[-1] for row in dic[value]])
        total_entropy -= ratio * ent

    return total_entropy


def build_tree(data, features):
    lastcol = [row[-1] for row in data]

    if len(set(lastcol)) == 1:
        node = Node("")
        node.answer = lastcol[0]
        return node

    n = len(data[0]) - 1
    gains = [0] * n

    for col in range(n):
        gains[col] = compute_gain(data, col)

    split = gains.index(max(gains))

    node = Node(features[split])

    fea = features[:split] + features[split + 1:]

    attr, dic = subtables(data, split, delete=True)

    for value in attr:
        child = build_tree(dic[value], fea)
        node.children.append((value, child))

    return node


def print_tree(node, level):
    if node.answer != "":
        print("\t" * level, node.answer)
        return

    print("\t" * level, node.attribute)

    for value, child in node.children:
        print("\t" * (level + 1), value)
        print_tree(child, level + 2)


def classify(node, x_test, features):
    if node.answer != "":
        print(node.answer)
        return

    pos = features.index(node.attribute)

    for value, child in node.children:
        if x_test[pos] == value:
            classify(child, x_test, features)
            return

    print("No matching branch found")


# Main program

base_dir = os.path.dirname(os.path.abspath(__file__))

dataset, features = load_csv(
    os.path.join(base_dir, "data3.csv")
)

# Remove Day because it is only an identifier
day_index = features.index("Day")

for row in dataset:
    del row[day_index]

features.remove("Day")

node1 = build_tree(dataset, features)

print("The decision tree for the dataset using ID3 algorithm is")
print_tree(node1, 0)


testdata, test_features = load_csv(
    os.path.join(base_dir, "data3_test.csv")
)

# Remove Day from test data
test_day_index = test_features.index("Day")

for row in testdata:
    del row[test_day_index]

test_features.remove("Day")

for xtest in testdata:
    print("The test instance:", xtest)
    print("The label for test instance:", end=" ")

    classify(node1, xtest, features)
