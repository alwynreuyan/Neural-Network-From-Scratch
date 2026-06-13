import numpy as np

inputs = [2, 1, 4, 5, 8]

weights = [[2, 1, 4, 5, 6],[2, 1, 4, 5, 6],[2, 1, 4, 5, 6],[2, 1, 4, 5, 6],[2, 1, 4, 5, 6]]
biases = [1,2,6,3,5]



output = np.dot(weights, inputs) +  biases
print(output)