import numpy as np
from sklearn.linear_model import LinearRegression

class Linear_regression():
	def __init__(self, x, y, theta0, theta1):
		self.x = x 
		self.y = y
		self.theta0 = theta0
		self.theta1 = theta1

	def object_function(self):
		return self.theta0 + self.theta1 * self.x

	def loss_function(self):
		return 0.5 * np.sum((self.y - self.object_function()) ** 2)

	def gradient_descent(self, alpha):
		d1 = np.sum((self.object_function() - self.y) * self.x)
		d0 = np.sum((self.object_function() - self.y))

		self.theta1 -= alpha * d1
		self.theta0 -= alpha * d0
		return self.theta1, self.theta0

	def standarize(self, x):
		mu = x.mean()
		std = x.std()
		self.x = (x - mu) / std
		


def main():
	diff = 1
	theta0 = theta1 = 1
	#Proof
	x = np.array([235, 216, 148, 35, 85, 204, 49, 25, 173, 191, 134, 99, 117, 112, 162, 272, 159, 159, 59, 198])
	y = np.array([591, 539, 413, 310, 308, 519, 325, 332, 498, 498, 392, 334, 385, 387, 425, 659, 400, 427, 319, 522])

	algorithm = Linear_regression(x, y, theta0, theta1)
	algorithm.standarize(x)
	error = algorithm.loss_function()
	while diff > 0.001:
		theta1, theta0 = algorithm.gradient_descent(1e-3)
		current_error = algorithm.loss_function()
		diff = error - current_error
		error = current_error
	print(theta1)
	
	mu = x.mean()
	std = x.std()
	x = (x - mu) / std

	model = LinearRegression()
	model.fit(x.reshape(-1, 1), y)
	print(model.coef_[0])

	x_b=np.c_[np.ones((x.shape[0],1)),x.reshape(-1, 1)]
	theta_best=np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
	print(theta_best[1])

if __name__ == '__main__':
	main()

