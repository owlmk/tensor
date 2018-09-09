import numpy as np 
import matplotlib.pyplot as plt
"""
x = 2 * np.random.rand(1000, 1)
y = 4 + 3*x + np.random.randn(1000, 1)

x_b = np.c_[np.ones((1000, 1)), x]

#print(x)

#print(x_b)

theba_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
print(y)
print(theba_best)

x_input = np.array([[0], [2]])
x_input_b = np.c_[(np.ones((2, 1))), x_input]

y_ = x_input_b.dot(theba_best)
print(y_)

plt.plot(x_input, y_, 'r-')
plt.plot(x, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()
"""

x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
x_b = np.c_[np.ones((100, 1)), x]

learning_rate = 0.1
n_iterations = 10000
m = 100

theta = np.random.randn(2, 1)
count = 0
for iteration in range(n_iterations):
	count += 1
	gradients = 1/m * x_b.T.dot(x_b.dot(theta)-y)
	theta = theta - learning_rate*gradients
	#print(gradients)

print(theta)
x_input = np.array([[0], [2]])
x_input_b = np.c_[(np.ones((2, 1))), x_input]

y_ = x_input_b.dot(theta)
print(x_input_b)