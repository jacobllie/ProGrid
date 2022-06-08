
import numpy as np

def crop(matrix,shape_diff):
    while len(matrix.shape) < 5:
        matrix = np.expand_dims(matrix, axis = 0)
    if shape_diff[0] % 2 != 0 and shape_diff[1] % 2 != 0:
        matrix_cropp = matrix[:,:,:,shape_diff[0]//2 + 1 : matrix.shape[3] - shape_diff[0]//2, shape_diff[1]//2 + 1: matrix.shape[4] - shape_diff[1]//2]
    elif shape_diff[0] % 2 != 0:
        matrix_cropp = matrix[:,:,:,shape_diff[0]//2 + 1: matrix.shape[3] - shape_diff[0]//2, shape_diff[1]//2: matrix.shape[4] - shape_diff[1]//2]
    elif shape_diff[1] % 2 != 0:
        matrix_cropp = matrix[:,:,:,shape_diff[0]//2 : matrix.shape[3] - shape_diff[0]//2, shape_diff[1]//2 + 1: matrix.shape[4] - shape_diff[1]//2]

    else:
        matrix_cropp = matrix[:,:,:,shape_diff[0]//2: matrix.shape[3] - shape_diff[0]//2, shape_diff[1]//2: matrix.shape[4] - shape_diff[1]//2]

    return matrix_cropp

def pad(matrix, shape_diff):
    if shape_diff[0] % 2 != 0 and shape_diff[1] % 2 != 0:
        matrix_pad = np.pad(matrix, ((shape_diff[0]//2 + 1, shape_diff[0]//2),(shape_diff[1]//2 + 1,shape_diff[1]//2)))

    elif shape_diff[0] % 2 != 0:
        matrix_pad = np.pad(matrix, ((shape_diff[0]//2 + 1, shape_diff[0]//2),(shape_diff[1]//2,shape_diff[1]//2)))
    elif shape_diff[1] % 2 != 0:
        matrix_pad = np.pad(matrix, ((shape_diff[0]//2, shape_diff[0]//2),(shape_diff[1]//2 + 1,shape_diff[1]//2)))

    else:
        matrix_pad = np.pad(matrix, ((shape_diff[0]//2, shape_diff[0]//2),(shape_diff[1]//2,shape_diff[1]//2)))

    return matrix_pad
