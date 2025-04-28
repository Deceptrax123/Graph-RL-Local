import pandas as pd
import numpy as np
import torch
import os
from dotenv import load_dotenv

load_dotenv('.env')


def remove_nans(data, markers):
    for i in range(1, markers+1):
        data["X"+str(i)] = data["X"+str(i)
                                ].fillna(data["X"+str(i)].mean())
        data["Y"+str(i)] = data["X"+str(i)
                                ].fillna(data["Y"+str(i)].mean())
        data["Z"+str(i)] = data["Z"+str(i)
                                ].fillna(data["Z"+str(i)].mean())

    return data


def create_data_matrix():
    data = pd.read_csv(os.getenv("MARKERS"))
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    num_markers = 39
    data_filled = remove_nans(data, num_markers)

    points = []
    for i in range(1, num_markers+1):
        col_x = data_filled["X"+str(i)]
        col_y = data_filled["Y"+str(i)]
        col_z = data_filled["Z"+str(i)]

        coordinates = np.stack((col_x, col_y, col_z))
        points.append(coordinates)

    final_points = np.stack(points)
    final_points_reshaped = np.transpose(final_points, (2, 0, 1))

    return final_points_reshaped


create_data_matrix()
