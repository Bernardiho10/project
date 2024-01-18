import numpy as np
from numpy.linalg import matrix_rank
from mpi4py import MPI
import os
import time
import matplotlib.pyplot as plt

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

comm.Barrier()

# Parameters
N = 10  # Number of users
D = 784  # Dimension of data (28x28 for MNIST)
U = N   # Assuming U is the same as N
q = 7
T = 0


# Function to generate random masks
def generate_random_mask(U, D):
    return np.random.rand(U, D)

# Function to encode sub-masks
def encode_sub_masks(mask, W_j):
    return np.dot(mask, W_j)

# Main LightSecAgg Procedure
def lightsecagg(N, U, D):
    W_j = np.random.rand(U, U)  # Example MDS matrix

    # Initialize encoded_masks to hold matrices for each user pair
    encoded_masks = np.empty((N, N, U, U))

    for i in range(N):
        z_i = generate_random_mask(U, D)
        for j in range(N):
            encoded_masks[i, j] = encode_sub_masks(z_i, W_j)

    # Phase 2: Masking and Uploading of Local Models
    local_models = np.random.rand(N, D)  # Placeholder for local models
    masked_models = local_models + np.sum(encoded_masks, axis=1)

    # Phase 3: Aggregate-Model Recovery
    U1 = np.random.choice(range(N), size=U, replace=False)
    aggregated_encoded_masks = np.zeros((U, U))  # Initialize with zeros
    for i in U1:
        # Sum over the appropriate axis to make the shapes compatible
        aggregated_encoded_masks += np.sum(encoded_masks[i], axis=0)

    # Ensure masked_models are correctly shaped for subtraction
    aggregated_model = np.zeros((U, D))  # Initialize with zeros
    for i in U1:
        # Adjusted to ensure shape alignment
       sum_masks = np.sum(aggregated_encoded_masks, axis=0).reshape(-1)
    aggregated_model += masked_models[i] - sum_masks

    # Check Rank Condition for Privacy with T=0
    if matrix_rank(aggregated_encoded_masks) == U:
        print("Privacy condition is satisfied")
        return np.sum(aggregated_model, axis=0) / U
    else:
        print("Privacy condition not satisfied. Abort aggregation process.")
        return None

# Execute the protocol
if rank == 0:
    aggregated_model = lightsecagg(N, U, D)
    if aggregated_model is not None:
        print("Aggregated Model:", aggregated_model)


# Integrated Functions from matrix_modification.py
def generate_zi(q, d):
    if q <= 1 or d <= 0:
        raise ValueError("q must > 1，d must > 0")
    z_i = np.random.randint(0, q, size=d)
    print("generate_zi output:", z_i)  # Debug print
    return z_i


def generate_MDS_matrix(U, N, q):

    if U <= 0 or N <= 0 or q < 1:
        raise ValueError("U和N必须大于0，q必须大于1")

    # 使用NumPy的random.randint函数生成U*N维矩阵，元素在0到q-1之间
    matrix = np.random.randint(0, q, size=(U, N))
    #print("generate_MDS_matrix output:", matrix)  # Debug print
    return matrix

def generate_pad_z(z_i, U, T):
#得到z和n填充之后的矩阵
    if U <= T:
        raise ValueError("U必须大于T")

    # 计算每份的列数
    cols_per_part = z_i.shape[0] // (U - T)

    # 使用NumPy的reshape函数将列向量分割成(U-T)份
    sub_matrices = np.array_split(z_i, U - T)

    # 将分割后的列向量按列组合成一个矩阵
    combined_matrix = np.column_stack(sub_matrices)
    cols_to_add = U - combined_matrix.shape[1]
    random_cols = np.random.randint(0, 10, size=(combined_matrix.shape[0], cols_to_add))
    padded_z = np.column_stack([combined_matrix, random_cols])
    print("generate_pad_z output:", padded_z)  # Debug print
    return padded_z

def generate_tilde_z(MDS_matrix, padded_z, U, T, N):
    tilde_z = generate_MDS_matrix(U, N, 1)
    #print(tilde_z)
    #print(f"tilde_z initial:\n{tilde_z}")

    for j in range(N):
        for i in range(U):
            tilde_z[:, j] = tilde_z[:, j] + padded_z[:, i] * MDS_matrix[i][j]
    return tilde_z

start_time = time.time()

MDS_matrix = generate_MDS_matrix(U,N,q)
z_i = generate_zi(q,D)
padded_z = generate_pad_z(z_i,U, T)
tilde_z = generate_tilde_z(MDS_matrix,padded_z,U, T, N)

end_time = time.time()
#elapsed_time = end_time - start_time
total_running_time = end_time - start_time 
print(f"Running time for a single round: {total_running_time} seconds")

print(f"tilde_z final:\n{tilde_z}")

if rank != 0:
    for i in range(1, N):
        if i != rank:
            #print(f"Before send: tilde_z[:, {i}] = {tilde_z[:, i]}")
            comm.send(tilde_z[:,i], dest=i)
if rank != 0:
    for i in range(1, N):
        if i != rank:
            recv_tilde_z=comm.recv(source=i)
            #print(f"After recv: tilde_z from rank {i} = {recv_tilde_z}")

if rank == 0:
    end_time = time.time()
    #running_time = end_time - start_time
    elapsed_time = end_time - start_time
    print("Process %d execution time: %.3f seconds" %(rank, elapsed_time))

# List of NPY file names
npy_files = ['local_beta_0.npy', 'local_beta_1.npy', 'local_beta_2.npy', 'local_beta_3.npy', 'local_beta_4.npy',
             'local_beta_5.npy', 'local_beta_6.npy', 'local_beta_7.npy', 'local_beta_8.npy', 'local_beta_9.npy',
             'local_beta_10.npy']

# Loop through each NPY file and read its data
for npy_file in npy_files:
    data = np.load('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset/local_beta_0.npy')
    data = np.load('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset/local_beta_1.npy')
    data = np.load('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset/local_beta_2.npy')
    data = np.load('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset/local_beta_3.npy')
    data = np.load('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset/local_beta_4.npy')
    data = np.load('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset/local_beta_5.npy')
    data = np.load('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset/local_beta_6.npy')
    data = np.load('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset/local_beta_7.npy')
    data = np.load('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset/local_beta_8.npy')
    data = np.load('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset/local_beta_9.npy')

    #Perform any necessary analysis on the data
    print("Analyzing", npy_file)
    print("Mean:", np.mean(data))
    print("Standard Deviation:", np.std(data))
    print("Minimum:", np.min(data))
    print("Maximum:", np.max(data))

    #Print a summary of the analysis
    print("----------------------------")

file_path = ('C:/Users/User/PycharmProjects/pythonProject/Blockchain-based-E-Voting-Simulation/dataset')
data = np.load(('./dataset/local_beta_0.npy'))
data = np.load(('./dataset/local_beta_1.npy'))
data = np.load(('./dataset/local_beta_2.npy'))
data = np.load(('./dataset/local_beta_3.npy'))
data = np.load(('./dataset/local_beta_4.npy'))
data = np.load(('./dataset/local_beta_5.npy'))
data = np.load(('./dataset/local_beta_6.npy'))
data = np.load(('./dataset/local_beta_7.npy'))
data = np.load(('./dataset/local_beta_8.npy'))
data = np.load(('./dataset/local_beta_9.npy'))

#np.save(file_path, data)

# Print the first few rows of the array
#print("Data Type:", data.dtype)
# Print the shape of the array to understand its dimensions
print("Shape of the array:", data.shape)
#print("First few rows of the array:\n", data[:5])
print(data, data[:10] )

# Once you understand the layout, you can extract the correct indices
# Here's an example assuming the number of users is in the first column and running time in the second
#users = data[:, 0]  # Replace 0 with the correct index for