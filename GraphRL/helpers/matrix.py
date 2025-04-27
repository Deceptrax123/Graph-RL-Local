import pandas as pd
import numpy as np
import torch
import os
from dotenv import load_dotenv

load_dotenv('.env')


def create_data_matrix():
    data = pd.read_csv(os.getenv("MARKERS"))
    num_markers = 39

    points = []
    for i in range(1, num_markers+1):
        col_x = data["X"+str(i)]
        col_y = data["Y"+str(i)]
        col_z = data["Z"+str(i)]

        coordinates = np.stack((col_x, col_y, col_z))
        points.append(coordinates)

    final_points = np.stack(points)
    final_points_reshaped = np.transpose(final_points, (2, 0, 1))

    print(final_points_reshaped[:, 0, 0])

    return final_points_reshaped
