input = [2, 1, 4, 5, 6]

weight1 = [1, 3, 4, 5, 3]
weight2 = [1, 3, 4, 5, 3]
weight3 = [1, 3, 4, 5, 3]
weight4 = [1, 3, 4, 5, 3]
weight5 = [1, 3, 4, 5, 3]
weight6 = [1, 3, 4, 5, 3]
weight7 = [1, 3, 4, 5, 3]

bias1 = 100
bias2 = 1
bias3 = 4
bias4 = 0
bias5 = 4
bias6 = 4
bias7 = -2

output = [[input[0]*weight1[0] + input[1]*weight1[1] + input[2]*weight1[2] + input[3]*weight1[3] + input[4]*weight1[4] + bias1],
		 [input[0]*weight2[0] + input[1]*weight2[1] + input[2]*weight2[2] + input[3]*weight2[3] + input[4]*weight2[4] + bias2]]


print(output)   