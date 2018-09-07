import numpy as np


x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

x_b = np.c_[np.ones((100, 1)), x]

n_epochs = 5000

t0, t1 = 5, 50
m = 100

def learn_schedule(t):
 	return t0 / (t + t1)




theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
	for i in range(m):
 		random_index = np.random.randint(m)
 		xi = x_b[random_index:random_index+1]
 		yi = y[random_index:random_index+1]
 		gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
 		learning_rate = learn_schedule(epoch*m + i)
 		theta = theta - learning_rate * gradients

print(theta)