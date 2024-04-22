import numpy as np


def pad_and_concatenate(tensor_list, axis=0, pad_value=0):
    # Step 1: Determine the maximum shape
    max_shape = tuple(max(s) for s in zip(*[tensor.shape for tensor in tensor_list]))

    # Step 2: Pad each tensor to match the maximum shape
    padded_tensors = []
    for tensor in tensor_list:
        pad_width = [(0, max_shape[i] - tensor.shape[i]) if i < len(tensor.shape) else (0, 0) for i in range(len(max_shape))]
        padded_tensor = np.pad(tensor, pad_width, mode='constant', constant_values=pad_value)
        padded_tensors.append(padded_tensor)

    # Step 3: Concatenate the padded tensors along the desired axis
    concatenated_tensor = np.concatenate(padded_tensors, axis=axis)

    return concatenated_tensor

# Example usage:
tensor_list = [np.array([[1, 2], [3, 4]]), np.array([[5, 6, 7], [8, 9, 10]]), np.array([[11, 12, 13, 14]])]
result = pad_and_concatenate(tensor_list, axis=0, pad_value=0)
print(result)