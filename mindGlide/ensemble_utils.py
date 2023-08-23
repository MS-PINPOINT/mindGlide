import nibabel as nib
import numpy as np
from scipy.stats import mode
import random
import string
import random


def custom_mode(arr):
    # custom mode ensures that ties are broken randomly
    unique_elements, counts = np.unique(arr, return_counts=True)
    max_count = np.max(counts)
    max_count_elements = unique_elements[counts == max_count]
    return random.choice(max_count_elements)


def generate_random_string(length):
    alphanumeric_chars = string.ascii_letters + string.digits
    return ''.join(random.choice(alphanumeric_chars) for _ in range(length))


def majority_vote(arrays):
    # Stack input arrays along a new axis
    stacked_arrays = np.stack(arrays, axis=-1)
    # Calculate the custom mode along the new axis and return the result
    result = np.apply_along_axis(custom_mode, axis=-1, arr=stacked_arrays)
    return result


def label_probabilities(arrays):
    '''
    The label_probabilities function calculates the probabilities 
    for each label at each index. It first stacks the input arrays 
    and finds the unique labels. Then, it initializes an array of 
    zeros with the same shape as the stacked arrays but with an additional 
    dimension for the unique labels.The function iterates through the unique 
    labels, counting occurrences of each label at each index and dividing by 
    the total number of input arrays. The result is an array of probabilities 
    for each label at each index.You can use these probabilities to make more 
    informed decisions about the final label, such as by applying a threshold or 
    considering other factors in your application.

    Args:
        arrays (list of numpy.ndarray): A list of 3D numpy arrays with shape (height, width, depth),
                                         each representing a set of labels for the same data.

    Returns:
        probabilities (numpy.ndarray): A 4D numpy array with shape (height, width, depth, num_unique_labels),
                                       where each element represents the probability of the corresponding
                                       label at the given index.
        unique_labels (numpy.ndarray): A 1D numpy array of unique labels found in the input arrays.

    Example:
        array1 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        array2 = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        array3 = np.array([[[0, 0, 3], [4, 5, 6], [7, 8, 9]]])

        probabilities, labels = label_probabilities([array1, array2, array3])

    '''
    stacked_arrays = np.stack(arrays, axis=-1)
    unique_labels = np.unique(stacked_arrays)
    probabilities = np.zeros(
        stacked_arrays.shape[:-1] + (len(unique_labels),), dtype=float)

    for idx, label in enumerate(unique_labels):
        label_count = np.sum(stacked_arrays == label, axis=-1)
        probabilities[..., idx] = label_count / stacked_arrays.shape[-1]

    return probabilities, unique_labels


def save_probabilities_nifti(probabilities, reference_nifti, output_file):
    """
    Save the probabilities as a 4D NIfTI file.

    Args:
        probabilities (numpy.ndarray): A 4D numpy array with shape (height, width, depth, num_unique_labels),
                                       representing the probabilities of each label at each index.
        reference_nifti (str): Path to a reference NIfTI file to copy the affine transformation and header.
        output_file (str): Path to save the output 4D NIfTI file.

    Example:
        probabilities = np.random.rand(30, 30, 30, 10)
        reference_nifti = "path/to/reference_nifti.nii.gz"
        output_file = "path/to/output_nifti.nii.gz"

        save_probabilities_nifti(probabilities, reference_nifti, output_file)
    """
    # Load the reference NIfTI file
    reference_img = nib.load(reference_nifti)

    # Create a new NIfTI image with the probabilities, affine transformation, and header from the reference image
    output_img = nib.Nifti1Image(
        probabilities, affine=reference_img.affine, header=reference_img.header)

    # Save the new NIfTI image to the output file
    nib.save(output_img, output_file)


def vanilla_majority_vote(all_labels):
    # Stack input arrays along a new axis
    stacked_arrays = np.stack(all_labels, axis=-1)

    # Calculate the mode along the new axis and return the result
    result, _ = mode(stacked_arrays, axis=-1, keepdims=True)
    return result.squeeze()
