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

def getEntropy(dataset):
    datasetLabels = dataset[:, -1] # get only the labels to calculate pk.
    _, occuranceArray = np.unique(datasetLabels, return_counts=True) # return the occurances of each sorted value(ascending).
    pkArray = occuranceArray/np.sum(occuranceArray) # return the true divison of each value in an array.
    entropy = -np.sum(pkArray * np.log2(pkArray)) # calculate the sum of each elementX element-wise multiplied with log2(elementX).
    return entropy

def getRemainder(subsetLeft, subsetRight):
    ssLeftSize, ssRightSize = len(subsetLeft), len(subsetRight) # get the amound of elements in the left and right subset
    ssAllSize = ssLeftSize + ssRightSize # get the sum of elements in both subsets
    leftValue = (ssLeftSize/ssAllSize) * getEntropy(subsetLeft) # : )
    rightValue = (ssRightSize/ssAllSize) * getEntropy(subsetRight) # : )
    remainder = leftValue + rightValue # calculate the remainder
    return remainder

def getInformationGain(dataset, subsetLeft, subsetRight):
    informationGain = getEntropy(dataset) - getRemainder(subsetLeft, subsetRight)
    return informationGain

class Node:
    def __init__(self, room=None, value=None, attribute=None, left=None, right=None):
        self.room = room
        self.value = value
        self.attribute = attribute
        self.left = left
        self.right = right

class Tree:
    def __init__(self, root=None):
        self.root = root


    def decision_tree_learning(self, training_dataset, depth):
        datasetLabels = training_dataset[:, -1] # split out so we get an 1-D array of all the labels
        countLabels = np.unique(datasetLabels) # count the number of unique labels and their occurances
        if len(countLabels) == 1: # if the length is == 1 we make a new tree node

            # print("\033[1;32m")
            # print("LEAFNODEFOUND_START")
            # print("\033[0m")
            # print(training_dataset)
            # print("\033[1;32m")
            # print("LEAFNODEFOUND_END")
            # print("\033[0m")

            return Node(room=countLabels[0]), depth # and return it here
        else:
            row, column = self.find_split(training_dataset) # remember, we need to sort the training set on the column it was retreived from.
            training_dataset = training_dataset[training_dataset[:,column].argsort()] # sort the tree (ascending) for the column used to split.
            tempNode = Node(value=training_dataset[row, column], attribute=column) # Make a new node and assign its right values.
            if not(self.root):
                self.root = tempNode

            # print("\033[1;33m")
            # print("Split into its LEFT part")
            # print("\033[0m")
            # print(training_dataset[:row, :])
            # print("\033[1;33m")
            # print("Split into its RIGHT part")
            # print("\033[0m")
            # print(training_dataset[row:, :])

            tempNode.left, l_depth = self.decision_tree_learning(training_dataset[:row, :], depth+1) # Recursion!
            tempNode.right, r_depth = self.decision_tree_learning(training_dataset[row:, :], depth+1)
            return tempNode, max(l_depth, r_depth) #return the tempNode up the chain.

    def find_split(self, training_dataset):

        finalIGValue = float('-inf') # ref values for outer loop. -inf just as a lower boundary.
        finalIGValueRow = float('-inf')
        finalIGValueColumn = float('-inf')

        # print("\033[1;35m")
        # print("Training set currently finding split of")
        # print("\033[0m")
        # print(training_dataset)

        for i in range(0, training_dataset.shape[1]-1):
            training_dataset = training_dataset[training_dataset[:,i].argsort()] # sort the dataset by current column (specified by i)

            for index in range(1, training_dataset.shape[0]):

                leftSubset = training_dataset[:index, :] # split the dataset into left and right
                rightSubset = training_dataset[index:, :]
                currentIG = getInformationGain(training_dataset, leftSubset, rightSubset) # see the IG it produces

                if currentIG > finalIGValue: # If this is true, we have found a better split than before
                    finalIGValue = currentIG
                    finalIGValueRow = index
                    finalIGValueColumn = i

        return finalIGValueRow, finalIGValueColumn # return the best split on the form of: row, column

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
    bestTree = None
    curAcc = float('-inf')

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
                root, depth = dTree.decision_tree_learning(trainingSet, 0)
                confusion = evaluate(valSet, root)
                acc = accuracy(confusion)
                print("Accuracy of current fold: ", acc, fold)
                if (acc > maxValAcc):
                    maxValAcc = acc
                    maxValAccTree = root
        testConfusion = evaluate(testSet, maxValAccTree)
        testAcc = accuracy(testConfusion) 
        print("Accuracy of current iteration: ", testAcc, iteration)

        if (testAcc > maxTestAcc):
            maxTestAcc = testAcc
            maxTestAccTree = maxValAccTree
            maxConfusion = testConfusion
        print("Max accuracy so far: ", maxTestAcc)

    

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

    print(confusion)
    return confusion

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


def main():
    #text = np.loadtxt("wifi_db/clean_dataset.txt") # set everything up and run.
    # dataSet, trainingSet, testSet = loadData("wifi_db/clean_dataset.txt", 42069)
    confusion, tree = crossValidation("wifi_db/clean_dataset.txt", 69, 5)

    print(accuracy(confusion))
    print(precision(confusion))
    print(recall(confusion))
    print(f1(confusion))

    # depth = 0
    # dTree = Tree()
    #root, depth = dTree.decision_tree_learning(text, depth)

# At execution look for name __main__ and run it.
if __name__ == "__main__":
    main()
