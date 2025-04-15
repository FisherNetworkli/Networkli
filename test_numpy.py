import numpy as np

# Create a simple array
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)

# Perform some basic operations
print("\nBasic operations:")
print("Mean:", np.mean(arr))
print("Sum:", np.sum(arr))
print("Standard deviation:", np.std(arr))

# Create a 2D array
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\n2D Matrix:")
print(matrix)

# Matrix operations
print("\nMatrix operations:")
print("Transpose:")
print(matrix.T)
print("\nMatrix multiplication:")
print(np.dot(matrix, matrix.T))

print("\nNumpy version:", np.__version__) 