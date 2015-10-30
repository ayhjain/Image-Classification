from learner import Learner
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import heapq
import csv


class TargetDistance(object): 

    def __init__(self, distance, value, classification):
        self.distance = distance
        self.value = value
        self.classification = classification

    def __lt__(self, other):
        return self.distance < other.distance

class BallTreeNode(object):

    def __init__(self, points, classification):
        self.is_leaf = (points.shape[0] == 1)
        if self.is_leaf:
            self.radius = 0
            self.pivot = points[0, :]
            self.classification = classification[0]
        else: 
            spread = points.max(0) - points.min(0)
            self.radius = spread.max() * 0.5
            self.pivot = 0.5 * (points.max(0) + points.min(0))
            self.classification = None
        self.child1 = None
        self.child2 = None

    def depth_print(self):
        if self.is_leaf:
            print self.pivot
        else:
            self.child1.depth_print()
            self.child2.depth_print()


    def min_distance(self, target):
        """ The minimum possible distance from the target to a point in this ball """
        difference = self.pivot - target
        return max(math.sqrt(np.dot(difference, difference)) - self.radius, 0)

    def distance(self, target):
        difference = self.pivot - target
        return math.sqrt(np.dot(difference, difference))

    def knn_search(self, target, size_k, heap):
        distance = self.min_distance(target)
        if self.is_leaf:
            if len(heap) < size_k:
                heapq.heappush(heap, TargetDistance(-distance, self.pivot, self.classification))
            else: 
                heapq.heappushpop(heap, TargetDistance(-distance, self.pivot, self.classification))
        elif len(heap) == size_k and distance >= -heap[0].distance: # compare against largest element in heap
            return
        else:
            closest, farthest = self.child1, self.child2
            if farthest.distance(target) < closest.distance(target):
                farthest, closest = self.child1, self.child2
            closest.knn_search(target, size_k, heap);
            farthest.knn_search(target, size_k, heap);


    @staticmethod
    def construct_balltree(points, classification):
        (n, m) = points.shape
        if n == 1:
            return BallTreeNode(points, classification)
        else: 
            spread = points.max(0) - points.min(0)
            c = spread.argmax()
            median = np.median(points[:, c])
            ball_tree = BallTreeNode(points, classification)
            left = np.ndarray((n/2, m))
            left_classification = np.ndarray((n/2))
            right = np.ndarray((n - n/2, m))
            right_classification = np.ndarray((n - n/2))
            left_index = 0
            right_index = 0
            same_as_median = []
            for i in range(n):
                if points[i, c] < median:
                    left[left_index, :] = points[i, :]
                    left_classification[left_index] = classification[i]
                    left_index += 1
                elif points[i, c] > median:
                    right[right_index, :] = points[i, :]
                    right_classification[right_index] = classification[i]
                    right_index += 1
                else:
                    same_as_median.append(i)
            for i in same_as_median:
                if left_index < n/2:
                    left[left_index, :] = points[i, :]
                    left_classification[left_index] = classification[i]
                    left_index += 1
                else:
                    right[right_index, :] = points[i, :]
                    right_classification[right_index] = classification[i]
                    right_index += 1
            ball_tree.child1 = BallTreeNode.construct_balltree(left, left_classification)
            ball_tree.child2 = BallTreeNode.construct_balltree(right, right_classification)
            return ball_tree



class KnnLearner(object):

    def __init__(self, train_x, train_y, k_nearest):
        """
            k_nearest is the number of neighbours used during lookup
        """
        (self.n, self.m) = train_x.shape
        self.k_nearest = k_nearest
        self.train_x = train_x
        self.train_y = train_y


    def get_vote(self, neighbours):
        """ 
        neighbours: a list of indices for the neighbours
        Returns the classification of the majority of neighbours 
        """
        votes = {}
        for i in neighbours:
            classification = self.train_y[i]
            if classification in votes:
                votes[classification] += 1
            else:
                votes[classification] = 1
        return max(votes, key=votes.get)
    


    def predict_with_tree(self, X, ball_tree):
        """ Predicts class for examples in X using knn and ball_tree for neighbour lookup"""
        predicted_y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            neighbours = []
            ball_tree.knn_search(X[i], self.k_nearest, neighbours)
            votes = {}
            for neighbour in neighbours:
                if neighbour.classification in votes:
                    votes[neighbour.classification] += 1
                else:
                    votes[neighbour.classification] = 1
            predicted_y[i] = max(votes, key=votes.get)
        return predicted_y

    def predict(self, X):
        """ Predicts class for example x using knn"""

        predicted_y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            neighbours = self.get_neighbours(X[i], self.k_nearest)
            predicted_y[i] = self.get_vote(neighbours)
        return predicted_y

    def get_neighbours(self, x, k):
        """ returns the indices of the k nearest neighbours to the example """
        k = min(k, self.n)
        nearest = {}
        for i in range(k):
            nearest[i] = self.euclidean_distance(x, self.train_x[i])
        for i in range(k, self.n):
            dist = self.euclidean_distance(x, self.train_x[i])
            if  dist < max(nearest.values()):
                nearest.pop(max(nearest, key=nearest.get))
                nearest[i] = dist
        return nearest

    def euclidean_distance(self, example_a, example_b):
        difference = example_a - example_b
        return math.sqrt(np.dot(difference, difference))

    def get_accuracy(self, predicted_y, actual_y, log_tests=False):
        """ Checks Accuracy of predictions """
        if log_tests:
            for i in range(actual_y.shape[0]):
                print 'predicted = {0}, actual = {1}'.format(predicted_y[i], actual_y[i])
        return float(sum(predicted_y == actual_y)) / predicted_y.shape[0]


if __name__=='__main__':
   
    # iris = datasets.load_iris()
    # X = iris.data[:, :2] # we only take the first two features.
    # y = iris.target
    # y[y == 2] = 1 # get rid of 3rd classifaction type

    with open('train_features.csv', 'r') as csvfile:
        data = list(csv.reader(csvfile))
        data = np.array(data)
        data = data.astype(np.float32)

    (n, m) = data.shape
    X = data[:, :m-1]
    y = data[:, m-1]
    perm = np.random.permutation(X.shape[0])
    X = X[perm, :]
    y = y[perm]
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    split = int(0.75 * X.shape[0])
    num_neighbours = 10
    train_x = X[0:split, :]
    train_y = y[0:split]
    test_x = X[split:, :]
    test_y = y[split:]

    tree = BallTreeNode.construct_balltree(train_x, train_y)

    learner = KnnLearner(train_x, train_y, num_neighbours)
    target = test_x[0, :]
    neighbours = learner.get_neighbours(target, num_neighbours)
    print '\nbrute force\n------------'
    for i in neighbours:
        print '{}, {}'.format(train_x[i], train_y[i])

    heap = []
    tree.knn_search(target, num_neighbours, heap)
    print '\nball tree\n------------'
    for x in heap:
        print '{}, {}'.format(x.value, x.classification)

    print '\ntarget'
    print target



    tree_predicted_y = learner.predict_with_tree(test_x, tree)
    predicted_y = learner.predict(test_x)
    print 'Accuracy of knn learner is {:.2%}'.format(learner.get_accuracy(predicted_y, test_y))
    print 'Accuracy of optimized knn learner is {:.2%}'.format(learner.get_accuracy(tree_predicted_y, test_y))


    plt.figure(2, figsize=(8, 6))
    plt.clf()

    results = ['black'] * test_x.shape[0]
    for i in range(len(results)):
        if predicted_y[i] == test_y[i]:
            results[i] = 'green'
    # Plot the training points    
    plt.scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap=plt.cm.Paired)
    # plt.scatter(test_x[:, 0], test_x[:, 1], c=results)
    plt.xlabel('music')
    plt.ylabel('movie')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

    


