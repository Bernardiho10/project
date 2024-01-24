import numpy as np
from mpi4py import MPI
import time
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
import os


# Function to load and partition the .npy dataset
def load_and_partition_data(num_users, filename='my_dataset.npy'):
    data = np.load(filename)
    partition_size = len(data) // num_users
    partitions = [data[i * partition_size:(i + 1) * partition_size] for i in range(num_users)]
    return partitions


# Simulated functions for encoding, decoding, and privacy verification
def encode_mask(mask, user_id, num_users):
    key = os.urandom(16)
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    encoded, _ = cipher.encrypt_and_digest(mask.tobytes())
    return encoded, nonce, key


def decode_aggregated_mask(encoded_masks, nonces, keys):
    decoded_masks = []
    for encoded, nonce, key in zip(encoded_masks, nonces, keys):
        cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
        decoded = cipher.decrypt(encoded)
        decoded_masks.append(np.frombuffer(decoded, dtype=np.int32))  # Adjust dtype as per your data
    return sum(decoded_masks) // len(decoded_masks)


def verify_privacy(encoded_masks, U):
    # Placeholder for privacy verification logic
    return True  # Temporarily bypassing the privacy check for testing purposes


def simulate_local_computation(partition):
    return np.mean(partition, axis=0)


def lightsecagg_simulation(partitions, rank, num_users):
    # MPI barrier for synchronization
    MPI.COMM_WORLD.Barrier()
    start_time = time.time()

    # Perform local computation
    local_result = simulate_local_computation(partitions[rank])

    # Introduce an artificial delay that increases with the process rank
    time.sleep(rank * 0.5)

    # MPI barrier for synchronization before starting communication
    MPI.COMM_WORLD.Barrier()

    # Simulate communication (gathering local results at root)
    all_results = MPI.COMM_WORLD.gather(local_result, root=0)

    # Barrier to ensure all processes have finished communication
    MPI.COMM_WORLD.Barrier()
    end_time = time.time()

    # Only root process performs aggregation
    if rank == 0:
        # Aggregate results (assuming aggregation is a summation)
        aggregate_result = sum(all_results)
        return end_time - start_time, aggregate_result
    else:
        return end_time - start_time, None



# Main function
def main():
    #MPI.Init()
    num_users = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    partitions = load_and_partition_data(num_users)

    # Run the simulation multiple times and take the average
    #num_runs = 5
    times = []
    for _ in range(num_users):
        execution_time, _ = lightsecagg_simulation(partitions, rank, num_users)
        times.append(execution_time)

    # Calculate the average execution time
    avg_time = sum(times) / num_users
    all_avg_times = MPI.COMM_WORLD.gather(avg_time, root=0)

    # Plotting is done only by the root process
    if rank == 0:
        plt.plot(range(num_users), all_avg_times, marker='o')
        plt.xlabel('Process Number')
        plt.ylabel('Average Execution Time (seconds)')
        plt.title('Ave8rage Execution Time vs. Process Number')
        plt.grid(True)
        plt.show()

    #MPI.Finalize()


if __name__ == "__main__":
    main()
