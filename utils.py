from cmath import pi
import numpy as np
from scipy.ndimage.filters import gaussian_filter


def get_hann_window(width: int, height: int) -> np.ndarray:
    win_row_1d = np.hanning(height)
    win_col_1d = np.hanning(width)

    return np.sqrt(np.outer(win_row_1d, win_col_1d))


def get_sub_window(frame: np.ndarray, center_pos: tuple, patch_size: tuple) -> np.ndarray:
    x1 = int(center_pos[0])-int((patch_size[0]-1)/2.0)
    y1 = int(center_pos[1])-int((patch_size[1]-1)/2.0)
    w = int(patch_size[0])
    h = int(patch_size[1])
    padding_left = 0
    padding_right = 0
    padding_top = 0
    padding_bot = 0

    # Deal with top and left exceeding image borders
    if (x1 < 0):
        padding_left = -x1
        x1 = 0
    if (y1 < 0):
        padding_top = -y1
        y1 = 0
    w -= padding_left
    h -= padding_top

    # Deal with right and bot exceeding image borders
    if (x1+w >= frame.shape[1]):
        padding_right = x1+w-frame.shape[1]
        w = frame.shape[1]-x1
    if (y1+h >= frame.shape[0]):
        padding_bot = y1+h-frame.shape[0]
        h = frame.shape[0]-y1

    # Crop image
    sub_win = frame[y1:y1+h, x1:x1+w]
    if (sub_win.ndim == 2):
        sub_win = np.pad(sub_win, [
                         (padding_top, padding_bot), (padding_left, padding_right)], mode='constant')
    elif (sub_win.ndim == 3):
        sub_win = np.pad(sub_win, [(padding_top, padding_bot),
                         (padding_left, padding_right), (0, 0)], mode='constant')

    return sub_win


def get_gaussian_response(width, height, sigma):
    response = np.zeros((height, width), np.float32)
    x_center = int((width-1)/2)
    y_center = int((height-1)/2)
    response[y_center, x_center] = 1.0

    # Gaussian peak centered at image
    response = gaussian_filter(response, sigma)

    # Move peak to top-left, with wrap around
    response = circshift(response, -int((width-1)/2), -int((height-1)/2))

    return response


def circshift(matrix: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if matrix.ndim == 2:
        nrows, ncols = matrix.shape
    elif matrix.ndim == 3:
        nrows, ncols, nchannels = matrix.shape
    else:
        raise Exception("Invalid number of dimensions")

    matrix_out = np.copy(matrix)
    for i in range(nrows):
        for j in range(ncols):
            idx_y = (i+dy) % nrows
            idx_x = (j+dx) % ncols
            matrix_out[idx_y, idx_x] = matrix[i, j]

    return matrix_out
