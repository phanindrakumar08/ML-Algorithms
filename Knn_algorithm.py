import numpy as np
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=1000, centers=2,
                  random_state=0, cluster_std=1.3)


def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i]-row2[i])**2
        return (distance)** 0.5



def nearest_neighbors(train, test_row, num_neighbors):
    distances = []
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors



def knn_prediction(train, test_row, num_neighbors):
    neigbors = nearest_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neigbors]
    prediction = max(set(output_values), key= output_values.count)
    return prediction

# Adding a column to the array using concatenate()
Z=np.concatenate([X, y.reshape(-1,1)], axis=1)
prediction = knn_prediction(Z, Z[0], 3)
print('Expected %d, Got %d.' % (y[0], prediction))