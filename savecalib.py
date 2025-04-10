import numpy as np
import pylab, scipy.io 

scipy.io.savemat('calibration_data1.mat', pylab.np.load('calibration_data.npz'))

with np.load("calibration_data.npz") as data:
    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']

    # Save data to a text file
    with open("calibration_data.txt", "w") as text_file:
        text_file.write("Camera Matrix:\n")
        text_file.write(np.array2string(camera_matrix, separator=', '))
        text_file.write("\n\nDistortion Coefficients:\n")
        text_file.write(np.array2string(dist_coeffs, separator=', '))

    # Save data to a MAT file
    scipy.io.savemat('calibration_data.mat', {
        'camera_matrix': camera_matrix,
        'dist_coeffs': dist_coeffs
    })