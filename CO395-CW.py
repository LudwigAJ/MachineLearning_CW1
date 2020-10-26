import numpy as np

def entropy(data):
    _, counts = np.unique(data, return_counts=True)
    pk = counts / sum(counts) 
    return -sum(pk * np.log2(pk))

def information_gain(sall, sleft, sright):
    remainder = (len(sleft)/len(sall)) * entropy(sleft) + (len(sright)/len(sall)) * entropy(sright)
    return entropy(sall) - remainder

class Tree:
    def __init__(self, root=None):
        self.root = root

    def decision_tree_learning(self, data, max_depth):
        def decision_tree_learning_rec(self, data, depth):
            if len(np.unique(data[:,-1])) == 1:
                return Node(data[0][-1]), depth
            else:
                attribute, value = self.find_split(data)
                return (attribute, value)

    def find_split(self, data):
        max_attr = 0
        max_value = float("-inf")
        max_ig = 0

        labels = data[:,-1]

        for attr_idx in range(len(data[0])-1):
            sorted_idx = np.argsort(data[:,attr_idx]) # This gives the sorted array via the element's old index
            sorted_labels = labels[sorted_idx] # This sorts the labels array withe the sorted_idx as the new indexs
            sorted_col = data[:,attr_idx][sorted_idx]
            
            cur_label = sorted_labels[0]

            for i in range(1,len(data)):
                if sorted_labels[i] != cur_label:
                    cur_label = sorted_labels[i]
                    sleft = sorted_col[:i]
                    sright = sorted_col[i:]
                    ig = information_gain(sorted_col, sleft, sright)

                    if ig > max_ig:
                        max_ig = ig
                        max_attr = attr_idx
                        max_value = sorted_col[i]
        return max_attr, max_value


class Node:
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value
        self.left = None
        self.right = None


class Leaf:
    pass


if __name__== "__main__":
    path= 'wifi_db/clean_dataset.txt'
    data = np.loadtxt(path)
    root = Tree()
    print(root.find_split(data))

    # print(data)