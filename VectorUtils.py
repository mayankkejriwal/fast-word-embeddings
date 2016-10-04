# Use this module to do various things with vectors e.g. normalize them. We do not claim they are efficient.
from sklearn.preprocessing import normalize
import warnings


def add_vectors(list_of_vectors):
    """
    None of the vectors in the list will be modified. We will raise an exception if the vectors turn out
    to be of unequal length.
    :param list_of_vectors:
    :return: an added vector
    """
    if not list_of_vectors:
        print 'Nothing to add. Returning None...'
        return None
    else:
        result = list(list_of_vectors[0])
        canonical_length = len(result)
        for i in range(1, len(list_of_vectors)):
            vector = list_of_vectors[i]
            if len(vector) != canonical_length:
                raise Exception('lengths of vectors are unequal. Exiting...')
            for j in range(0, canonical_length):
                result[j] += vector[j]
        return result


def normalize_vector(vector):
    """
    l2-normalizer
    :param vector:
    :return: A normalized vector. Original vector is not modified.
    """
    warnings.filterwarnings("ignore")
    return normalize(vector)[0]


def normalize_matrix(matrix):
    """
    l2-normalizer
    :param matrix: use numpy for building this
    :return: A normalized matrix. original is not modified
    """
    warnings.filterwarnings("ignore")
    return normalize(matrix)


def non_zero_element_fraction(vector):
    """
    Returns fraction of elements in vector that is non-zero
    :param vector:
    :return: A float
    """
    num = 0.0
    for v in vector:
        if v != 0:
            num += 1.0
    return num/len(vector)