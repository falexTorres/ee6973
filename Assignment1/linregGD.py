import numpy as np

X = np.array([0, 1.0, 2.0, 3.0, 4.0])
Y = np.array([1.0, 3.0, 7.0, 13.0, 21.0])
m = 0
B = np.zeros(5)

Yhat = m * X + B
error = Yhat - Y
step = 0.14205

iterations = 10000
count = 0

for j in range(iterations):
	count += 1
	partial_m = 0
	partial_b = 0

	# compute partial of error with respect to slope
	for i in range(len(X)):
		partial_m += (2*X[i]*(m*X[i] + B[i] - Y[i])) / len(B)

	# compute partial of error with respect to intercept
	for i in range(len(X)):
		partial_b += (2*(m*X[i] + B[i] - Y[i])) / len(B)

	# compute the adjustment as step size times derivative
	b = step*partial_b
	for i in range(len(B)):
		B[i] -= b 

	m -= step*partial_m
	gradient_magnitude = ((partial_b**2) + (partial_m**2))**(0.5)
	if (gradient_magnitude**(0.5) < 0.00001):
		 break

print("regression finished in", count, "iterations.")
print("y = ", m, "x + ", B[0])
print("mean error:", gradient_magnitude**(0.5))
