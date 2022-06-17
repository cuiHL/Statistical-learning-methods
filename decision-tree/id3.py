import math
import numpy as np
from sklearn.preprocessing import label_binarize

class Node(object):
    def __init__(self, data=None):
        self.data = data
        self.child = {}

class DecisionTree(object):

    def create(self, dataset, labels, option='ID3'):
        """
        :param dataset
        :param labels
        :param option
        :return: a root node
        """

        feature_column = self.take_feature_value(dataset)
        print(feature_column)

        def create_branch(dataset, feature_column):

            label = [row[-1] for row in dataset]
            #建立根节点
            node = Node()
            if len(set(label)) == 1:
                node.data = label[0]
                node.child = None
                return node
            
            entropy = self.entropy(dataset)
    
            #定义信息增益
            max_entropy_gain = 0
            #定义信息增益的索引
            max_entropy_index = 0
            
            #取出熵最大增益的列id
            for feature in feature_column.keys():
                if option == 'ID3':
                    entropy_gain = entropy - self.conditional_entropy(dataset, feature)    
                else:
                    entropy_gain = (entropy - self.conditional_entropy(dataset, feature)) / entropy
                
                if entropy_gain >= max_entropy_gain:
                    max_entropy_gain = entropy_gain
                    max_entropy_index = feature
            
            #将特征列复制
            feature_column_dup = feature_column.copy()
            node.data = labels[max_entropy_index]
            #删除已划分过的特征
            del feature_column_dup[max_entropy_index]
    
            #继续切分子集
            for feature_value in feature_column[max_entropy_index]:
                sub_dataset = [row for row in dataset if row[max_entropy_index] == feature_value]
                node.child[feature_value] = create_branch(sub_dataset, feature_column_dup)
        
            return node 
        
        return create_branch(dataset, feature_column) 


    def classify(self, node, description, sample):
        """
        :param node: tree root node
        :param description: 
        :param sample: test
        """
        while node != None:
            if node.data in description:
                index = description.index(node.data)
                #取出值
                x = sample[index]
                for key in node.child:
                    if x == key:
                        node = node.child[key]
            else:
                break 
        return node.data

    def take_feature_value(self, dataset):

        """
        :param dataset
        :return dict
        """

        feature_column = {}

        _, n = np.shape(dataset)
        for i in range(n - 1):
            column = list(set([row[i] for row in dataset]))
            feature_column[i] = column
        
        return feature_column 

    def conditional_entropy(self, dataset, feature_key):
        """
        :param dataset
        :param feature_key
        :return float
        """
        #取出特征列
        feature_column = [row[feature_key] for row in dataset]
        condition_entropy = 0
        for feature in set(feature_column):
            sub_dataset = [row for row in dataset if row[feature_key] == feature]
            condition_entropy += (feature_column.count(feature)/ float(len(feature_column))) * self.entropy(sub_dataset)
        return condition_entropy

    #计算熵
    def entropy(self, dataset, feature_key=-1):
        """
        :param dataset
        :param calculate entropy based on feature key
        :return float
        """
        #取出label列
        feature_column = [row[feature_key] for row in dataset]
        entropy = 0
        for label in list(set(feature_column)):
            p = feature_column.count(label) / float(len(feature_column))
            entropy -= p * math.log(p, 2)
        return entropy

if __name__ == '__main__':
    dataset = [['青年', '否', '否', '一般', '否'],
           ['青年', '否', '否', '好', '否'],
           ['青年', '是', '否', '好', '是'],
           ['青年', '是', '是', '一般', '是'],
           ['青年', '否', '否', '一般', '否'],
           ['中年', '否', '否', '一般', '否'],
           ['中年', '否', '否', '好', '否'],
           ['中年', '是', '是', '好', '是'],
           ['中年', '否', '是', '非常好', '是'],
           ['中年', '否', '是', '非常好', '是'],
           ['老年', '否', '是', '非常好', '是'],
           ['老年', '否', '是', '好', '是'],
           ['老年', '是', '否', '好', '是'],
           ['老年', '是', '否', '非常好', '是'],
           ['老年', '否', '否', '一般', '否']]
    
    description = ['年龄', '有工作', '有自己的房子', '信贷情况']
    tree = DecisionTree()
    node = tree.create(dataset, description)

    for line in dataset:
        print(tree.classify(node, description, line))