"""
DESCRIPTION
    A house has 7 WIFI hotspots pumping out Wi-Fi. Based on the signal strength of each one, we need to predict which room you are in.
    We are given a 2000x8 array in .txt format. Where each column corresponds to the Wi-Fi strength originating from that hotspot.
    i.e. The left-most column shows the strength a person is receiving from that specific hotspot (in dB I assume).
    The right-most column states which room the person was in that instance.
IMPLEMENTATION
    Given that we have 7-attributes to choose from, we need to sort through each one that will give us the best IG.
    We will have two custom types. One Node and one Tree. A Tree will simply contain Nodes starting from a root.
    The nodes will have a value to compare against which is the Wi-Fi hotspot we're currently using for comparison [0, 6].
    They will also have an an attribute called attribute which says which Wi-Fi hotspot we're comparing. i.e. the index [0. 6].
    Each node will also have a Room attribute. This is usually = None, but if it is not None, we have found a room.
    Each node will also have its two children defined as left and right. Of which they can be Node or None.
    When going through the tree at a later stage, I guess we just compare the value from its attribute at a given node
    and if (it is less than or greater than?) I cannot think of which currently, but I think if it is less than the value at the node
    we go left node and if not we go right. Or maybe it was vice versa lmao.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

def getEntropy(dataset):
    datasetLabels = dataset[:, -1] # get only the labels to calculate pk.
    _, occuranceArray = np.unique(datasetLabels, return_counts=True) # return the occurances of each sorted value(ascending).
    pkArray = occuranceArray/np.sum(occuranceArray) # return the true divison of each value in an array.
    entropy = -np.sum(pkArray * np.log2(pkArray)) # calculate the sum of each elementX element-wise multiplied with log2(elementX).
    return entropy

def getRemainder(subsetLeft, subsetRight):
    ssLeftSize, ssRightSize = len(subsetLeft), len(subsetRight) # get the amound of elements in the left and right subset
    ssAllSize = ssLeftSize + ssRightSize # get the sum of elements in both subsets
    leftValue = (ssLeftSize/ssAllSize) * getEntropy(subsetLeft) 
    rightValue = (ssRightSize/ssAllSize) * getEntropy(subsetRight) 
    remainder = leftValue + rightValue # calculate the remainder
    return remainder

def getInformationGain(dataset, subsetLeft, subsetRight):
    informationGain = getEntropy(dataset) - getRemainder(subsetLeft, subsetRight)
    return informationGain

class Node:
    def __init__(self, room=None, value=None, attribute=None, left=None, right=None, size=0):
        self.room = room
        self.value = value
        self.attribute = attribute
        self.left = left
        self.right = right
        self.size = size
        
    def display(self):
        lines, *_ = self._display_aux()
        for line in lines:
            print(line)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.right is None and self.left is None:
            line = '%s' % self.room
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.right is None:
            lines, n, p, x = self.left._display_aux()
            s = '%s' % self.room
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.left is None:
            lines, n, p, x = self.right._display_aux()
            s = '%s' % self.room
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.left._display_aux()
        right, m, q, y = self.right._display_aux()
        s = '%s' % self.room
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

class Tree:
    def __init__(self, root=None):
        self.root = root

    def decisionTreeLearning(self, trainingDataset, depth):
        datasetLabels = trainingDataset[:, -1] # split out so we get an 1-D array of all the labels
        countLabels = np.unique(datasetLabels) # count the number of unique labels and their occurances
        if len(countLabels) == 1: # if the length is == 1 we make a new tree node

            # print("\033[1;32m")
            # print("LEAFNODEFOUND_START")
            # print("\033[0m")
            # print(trainingDataset)
            # print("\033[1;32m")
            # print("LEAFNODEFOUND_END")
            # print("\033[0m")

            return Node(room=countLabels[0], size=len(countLabels)), depth # and return it here
        else:
            row, column = self.findSplit(trainingDataset) # remember, we need to sort the training set on the column it was retreived from.
            trainingDataset = trainingDataset[trainingDataset[:,column].argsort()] # sort the tree (ascending) for the column used to split.
            tempNode = Node(value=trainingDataset[row, column], attribute=column) # Make a new node and assign its right values.
            if not(self.root):
                self.root = tempNode

            # print("\033[1;33m")
            # print("Split into its LEFT part")
            # print("\033[0m")
            # print(trainingDataset[:row, :])
            # print("\033[1;33m")
            # print("Split into its RIGHT part")
            # print("\033[0m")
            # print(trainingDataset[row:, :])

            tempNode.left, leftDepth = self.decisionTreeLearning(trainingDataset[:row, :], depth+1) # Recursion!
            tempNode.right, rightDepth = self.decisionTreeLearning(trainingDataset[row:, :], depth+1)
            return tempNode, max(leftDepth, rightDepth) #return the tempNode up the chain.

    def findSplit(self, trainingDataset):

        finalIGValue = float('-inf') # ref values for outer loop. -inf just as a lower boundary.
        finalIGValueRow = float('-inf')
        finalIGValueColumn = float('-inf')

        # print("\033[1;35m")
        # print("Training set currently finding split of")
        # print("\033[0m")
        # print(trainingDataset)

        for i in range(0, trainingDataset.shape[1]-1):
            trainingDataset = trainingDataset[trainingDataset[:,i].argsort()] # sort the dataset by current column (specified by i)

            for index in range(1, trainingDataset.shape[0]):

                leftSubset = trainingDataset[:index, :] # split the dataset into left and right
                rightSubset = trainingDataset[index:, :]
                currentIG = getInformationGain(trainingDataset, leftSubset, rightSubset) # see the IG it produces

                if currentIG > finalIGValue: # If this is true, we have found a better split than before
                    finalIGValue = currentIG
                    finalIGValueRow = index
                    finalIGValueColumn = i

        return finalIGValueRow, finalIGValueColumn # return the best split on the form of: row, column

    def pruneTree(self, validationSet):
        def postOrderTraversal(node=self.root):
            if node:
                if node.room: return
                postOrderTraversal(node.left)
                postOrderTraversal(node.right)

                # check left is leaf, right is leaf, this node is not leaf
                canPrune = node.left.room and node.right.room
                if canPrune:
                    # compare validation error
                    # evalute with current node first
                    unprunedConfusion = evaluate(validationSet, self.root)
                    unprunedAcc = accuracy(unprunedConfusion) # this is the validation error
                    
                    # prune
                    old_room, old_size = node.room, node.size
                    # obtain majority of left and right
                    if node.left.size <= node.right.size:
                        node.size = node.right.size
                        node.room = node.right.room # becomes leaf node
                    else:
                        node.size = node.left.size
                        node.room = node.left.room
                    
                    prunedConfusion = evaluate(validationSet, self.root)
                    prunedAcc = accuracy(prunedConfusion) # this is the validation error

                    # reset node to previous copy
                    if unprunedAcc > prunedAcc:
                        node.room, node.size = old_room, old_size
                    else:
                        node.right = node.left = None
        return postOrderTraversal()

    def getDepth(self):
        def rec(node):
            if node.room:
                return 0
            return 1 + max(rec(node.left), rec(node.right))
        return rec(self.root)
def basicLoading(datasetPath, seed):
    dataSet = np.loadtxt(datasetPath)
    np.random.seed(seed)

    #Get the different indices
    dataSetIndices = np.arange(len(dataSet))
    trainingSetIndices = np.random.choice(len(dataSet), int(0.9*len(dataSet)), replace=False)
    testSetIndices = np.delete(dataSetIndices, trainingSetIndices)
    #Slice the numpy arrays
    trainingSet = dataSet[trainingSetIndices, :]
    testSet = dataSet[testSetIndices, :]

    return dataSet, trainingSet, testSet

def crossValidation(datasetPath, seed, k):

    dataSet = np.loadtxt(datasetPath)
    np.random.seed(seed)
    np.random.shuffle(dataSet)
    foldSize = len(dataSet)//k

    dataSetIndices = np.arange(len(dataSet))

    maxTestAcc = 0
    maxTestAccTree = None
    maxConfusion = None

    for iteration in range(k):
        testSetIndices = dataSetIndices[iteration*foldSize : (iteration+1)*foldSize]
        testSet = dataSet[testSetIndices, :]

        maxValAcc = 0
        maxValAccTree = None
        for fold in range(k):
            if iteration != fold:
                valSetIndices = dataSetIndices[fold*foldSize : (fold+1)*foldSize]
                valSet = dataSet[valSetIndices, :]
                trainingSetIndices = np.delete(dataSetIndices, np.hstack([testSetIndices, valSetIndices]))
                trainingSet = dataSet[trainingSetIndices, :]

                dTree = Tree()
                root, depth = dTree.decisionTreeLearning(trainingSet, 0)
                # dTree.root.display()
                print("Max depth before pruning:", depth)
                dTree.pruneTree(valSet)
                prunedDepth = dTree.getDepth()
                print("Max depth after pruning:", prunedDepth)
                # dTree.root.display()
                
                confusion = evaluate(valSet, root)
                acc = accuracy(confusion)

                print("Accuracy of current fold (validation set): ", acc, fold)
                if (acc > maxValAcc):
                    maxValAcc = acc
                    maxValAccTree = root
        testConfusion = evaluate(testSet, maxValAccTree)
        testAcc = accuracy(testConfusion) 
        print("Accuracy of current iteration (test set): ", testAcc, iteration)

        if (testAcc > maxTestAcc):
            maxTestAcc = testAcc
            maxTestAccTree = maxValAccTree
            maxConfusion = testConfusion
        print("Max accuracy so far (test set): ", maxTestAcc)

    

    return maxConfusion, maxTestAccTree

def evaluate(data, trainedTree):
    preds = []
    labels = data[:,-1]
    uniqueLabels = len(np.unique(labels))
    confusion = np.zeros((uniqueLabels,uniqueLabels))

    for sample in data:
        node = trainedTree
        while (node.room is None):
            if sample[node.attribute] < node.value:
                node = node.left
            else:
                node = node.right

        preds.append(node.room)

    
    for i in range(len(labels)):
        confusion[int(labels[i]-1), int(preds[i]-1)] += 1

    return confusion

# Metrics
def accuracy(confusion):
    correct = 0
    for i in range(len(confusion)):
        correct += confusion[i, i]
    total = confusion.sum()

    return correct / total

def precision(confusion):
    output = []
    for i in range(len(confusion)):
        correct = confusion[i, i]
        total = confusion[:, i].sum()
        output.append(correct/total)

    return output

def recall(confusion):
    output = []
    for i in range(len(confusion)):
        correct = confusion[i, i]
        total = confusion[i, :].sum()
        output.append(correct/total)

    return output

def f1(confusion):
    precisionArr, recallArr = precision(confusion), recall(confusion)
    precisionAvg, recallAvg = np.mean(precisionArr), np.mean(recallArr)
    output = 2 * (precisionAvg*recallAvg) / (precisionAvg+recallAvg)

    return output

def heatmap(confusion, cbar_kw={}, **kwargs):
    fig, ax = plt.subplots()
    im = ax.imshow(confusion, **kwargs)



    # We want to show all ticks...
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 2, 3])
    # ... and label them with the respective list entries
    ax.set_xticklabels(['Room 1', 'Room 2', 'Room 3', 'Room 4' ])
    ax.set_yticklabels(['Room 1', 'Room 2', 'Room 3', 'Room 4' ])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")


    # Loop over data dimensions and create text annotations.
    for i in range(4):
        for j in range(4):
            text = ax.text(j, i, confusion[i, j],
                        ha="center", va="center", color="black")

    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel("counts", rotation=-90, va="bottom")

    ax.set_title("Clean Dataset Confusion Matrix")
    fig.tight_layout()
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.show()




def main():
    #text = np.loadtxt("wifi_db/clean_dataset.txt") # set everything up and run.
    # dataSet, trainingSet, testSet = loadData("wifi_db/clean_dataset.txt", 42069)
    confusion, tree = crossValidation("wifi_db/clean_dataset.txt", 69, 10)

    # confusion = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])

    heatmap(confusion, cmap="Blues")
    # print(accuracy(confusion))
    # print(precision(confusion))
    # print(recall(confusion))
    # print(f1(confusion))
    # tree.display()

# At execution look for name __main__ and run it.
if __name__ == "__main__":
    main()
