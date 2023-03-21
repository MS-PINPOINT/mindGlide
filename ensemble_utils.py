def majority_vote(all_labels):
            # Stack input arrays along a new axis
            stacked_arrays = np.stack(all_labels, axis=-1)

            # Calculate the mode along the new axis and return the result
            result, _ = mode(stacked_arrays, axis=-1)
            return result.squeeze()