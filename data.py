import numpy as np

# Example: Create a dataset of random numbers
data = np.random.rand(1000, 784)  # 1000 samples, 784 features

# Save this data to a .npy file
np.save('my_dataset.npy', data)
