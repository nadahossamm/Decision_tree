import pandas as pd
import numpy as np
import math
import statistics as sc

maxDepth = 0

def completing_data(X):
    for i in headerList:
        column = X[i].tolist()
        yes = column.count('y')
        no = column.count('n')
        for j in range(len(column)):
            if column[j] == '?' and yes >= no:
                column[j] = 'y'
            elif column[j] == '?' and no > yes:
                column[j] = 'n'
        X[i] = column

    return X


def calculate_output_probabilities(X):
    column = X['Class Name'].tolist()
    R = 0
    D = 0
    for i in range(len(column)):
        if column[i] == 'republican':
            R = R + 1
        else:
            D = D + 1
    prob_r = R / len(column)
    prob_d = D / len(column)

    return prob_r, prob_d


def calculate_more_party(X):
    output = list(X['Class Name'])
    r = 0
    d = 0
    for i in range(len(output)):
        if output[i] == 'democrat':
            d += 1
        else:
            r += 1
    if r > d:
        return 'republican'
    else:
        return 'democrat'


def calculate_information_gain(X):
    party = X['Class Name'].tolist()
    informationGains = []
    a, b = calculate_output_probabilities(X)
    if a == 0 or b == 0:
        return 1, 0
    entropy_parent = calculate_entropy(a, b)
    headers = list(X.columns)
    for i in range(1, len(headers)):
        yesRCount = 0
        yesDCount = 0
        noRCount = 0
        noDCount = 0
        column = X[headers[i]].tolist()
        for j in range(len(column)):
            if column[j] == 'y' and party[j] == 'republican':
                yesRCount += 1
            elif column[j] == 'y' and party[j] == 'democrat':
                yesDCount += 1
            elif column[j] == 'n' and party[j] == 'republican':
                noRCount += 1
            elif column[j] == 'n' and party[j] == 'democrat':
                noDCount += 1
        if (noDCount + noRCount) == 0 or (yesDCount + yesRCount) == 0:
            information_gain = 0
        else:

            yesD = yesDCount / (yesDCount + yesRCount)
            yesR = yesRCount / (yesDCount + yesRCount)

            if (yesD == 0 and yesR != 0) or (yesD != 0 and yesR == 0):
                entropy_y = 0
            else:
                entropy_y = calculate_entropy(yesD, yesR)
            noD = noDCount / (noDCount + noRCount)
            noR = noRCount / (noDCount + noRCount)

            if (noD == 0 and noR != 0) or (noD != 0 and noR == 0):
                entropy_n = 0
            else:
                entropy_n = calculate_entropy(noD, noR)

            information_gain = entropy_parent - ((((yesDCount + yesRCount) / (len(column))) * entropy_y) + (
                        ((noDCount + noRCount) / (len(column))) * entropy_n))
        informationGains.append(information_gain)

    return informationGains.index(max(informationGains)) + 1, max(informationGains)


# bna5od el dataset kolaha w el attribute ele ha-split 3ando (y/n)
# w bta5od el parent ele ana gai meno w btraga3 el goz2 ele fel dataset ele feeh
# el parent bysawe el attribute ele etba3at (y/n)
def get_splitted_dataframe(x, attribute, parent):
    h = list(x[parent])
    tmp = np.array(x)
    y = []

    for i in range(len(h)):
        if h[i] == attribute:
            y.append(tmp[i])

    y = pd.DataFrame(y, columns=x.columns)

    return y


def calculate_entropy(prob_a, prob_b):
    entropy = -(prob_a * math.log(prob_a, 2)) - (prob_b * math.log(prob_b, 2))
    return entropy


def is_set_pure(X, party, y_or_n, column_name):
    outputs = list(X['Class Name'])
    curr_column = list(X[column_name])
    for i in range(len(curr_column)):
        if (curr_column[i] == y_or_n) and (outputs[i] != party):
            return False
    return True


def build_tree(node, x, header):
    index, info_gain = calculate_information_gain(x)

    if info_gain == 0:
        node.political_party = calculate_more_party(x)
        return

    node.political_party = header[index]

    rightSide = get_splitted_dataframe(x, 'y', header[index])
    node.right = Node(None, None, "")
    build_tree(node.right, rightSide, header)

    leftSide = get_splitted_dataframe(x, 'n', header[index])
    node.left = Node(None, None, "")
    build_tree(node.left, leftSide, header)


def predicted_party(X_T, root, headers):
    curr = root

    while curr.political_party != 'republican' and curr.political_party != 'democrat':
        # print(curr.political_party)
        i = headers.index(curr.political_party)
        if X_T[i] == 'y':
            curr = curr.right
        elif X_T[i] == 'n':
            curr = curr.left
    return curr.political_party


def calculate_accuracy(X_t, tree_root, header):
    x_temp = np.array(X_t)
    right = 0
    wrong = 0

    for i in range(X_t.shape[0]):
        predict = predicted_party(x_temp[i], tree_root, header)
        if predict == x_temp[i][0]:
            right += 1
        else:
            wrong += 1
    calcAccuracy = (right / X_t.shape[0]) * 100
    return calcAccuracy


class Node:
    def __init__(self, left, right, political_party):
        self.left = left
        self.right = right
        self.political_party = political_party

    def depth(self):
        if self.left == None and self.right == None:
            return 1
        elif self.left == None:
            return self.right.depth() + 1
        elif self.right == None:
            return self.left.depth() + 1
        else:
            return max(self.left.depth(), self.right.depth()) + 1


class tree:
    def __init__(self, root):
        self.root = None

    def traverse(self, node):
        if node == None:
            return
        self.traverse(node.right)
        self.traverse(node.left)

        print(node.political_party)


    def calculateTreeSize(self, node):
        if node == None:
            return 0

        return self.calculateTreeSize(node.right) + self.calculateTreeSize(node.left) + 1


def splitData(dataset, ratio):
    shuffle_df = dataset.sample(frac=1)
    train_size = int(ratio * len(dataset))
    X_train = shuffle_df[:train_size]
    X_test = shuffle_df[train_size:]
    pd.DataFrame(X_train, columns=headerList)
    pd.DataFrame(X_test, columns=headerList)
    return X_train, X_test


headerList = ['Class Name', 'handicapped-infants', 'water-project-cost-sharing', 'adoption-of-the-budget-resolution',
              'physician-fee-freeze',
              'el-salvador-aid', 'religious-groups-in-schools', 'anti-satellite-test-ban', 'aid-to-nicaraguan-contras',
              'mx-missile',
              'immigration', 'synfuels-corporation-cutback', 'education-spending', 'superfund-right-to-sue', 'crime',
              'duty-free-exports', 'africa']

read_file = pd.read_csv("house-votes-84.data.txt", sep=",", names=headerList)
read_file = completing_data(read_file)
df = pd.DataFrame()

# d_tree.traverse(root)
# print(d_tree.updateTreeSize(root))

# print(calculate_accuracy(X_test,root,headerList))
accuracies = []
treeDepth = []
noNode =[]
for i in range(5):
    X_train, X_test = splitData(read_file, 0.25)

    info_counter = 0
    a, b = calculate_output_probabilities(X_train)

    root = Node(None, None, '')

    build_tree(root, X_train, headerList)
    d_tree = tree(root)

    accuracyy = calculate_accuracy(X_test, root, headerList)
    tree_depth = root.depth()
    noNode.append(d_tree.calculateTreeSize(root))
    accuracies.append(accuracyy)
    treeDepth.append(tree_depth)

df = pd.DataFrame()
df['accuracies'] = accuracies
df['treeDeapth'] = treeDepth
df['treesize'] = noNode

df.to_excel('first 5 Runs .xlsx', index=False)

accuracy = []
treeSize = []
maxacc = []
minacc = []
meanacc = []
maxtree = []
mintree = []
meantree = []
df = pd.DataFrame()

ranges = [0.30, .40, .50, .60, .70]
for i in ranges:
    accuracy = []
    treeSize = []
    for j in range(5):
        X_train, X_test = splitData(read_file, i)
        info_counter = 0
        a, b = calculate_output_probabilities(X_train)

        root = Node(None, None, '')

        build_tree(root, X_train, headerList)
        d_tree = tree(root)

        accuracyy = calculate_accuracy(X_test, root, headerList)
        tree_size = root.depth()
        accuracy.append(accuracyy)
        treeSize.append(tree_size)

    maxacc.append(max(accuracy))
    minacc.append(min(accuracy))
    meanacc.append(sc.mean(accuracy))
    maxtree.append(max(treeSize))
    mintree.append(min(treeSize))
    meantree.append(sc.mean(treeSize))

df["Traning set Size"] = ranges
df["Max accuracy"] = maxacc
df["Min accuracy"] = minacc
df["mean accuracy"] = meanacc
df["max Tree depth"] = maxtree
df["min Tree depth"] = mintree
df["mean Tree depth"] = meantree

df.to_excel('runs.xlsx', index=False)
