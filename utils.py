import numpy as np


def unpack_data(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpacks the data as given in the files *.data*_cmb.txt.

    :param filename: the filename
    :type filename: str
    :return: returns the angles and powers
    in the txt file as a tuple of np.ndarrays
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    angles = []
    powers = []
    with open(filename, "r") as file:
        for line in file:
            deg_power_list = line.strip().split(
                " "
            )  # Removes trailing whitespace and newlines
            angles.append(deg_power_list[0])
            powers.append(deg_power_list[1])
    powers.pop(0)
    angles.pop(0)

    powers_floats = [float(val) for val in powers]
    angles_floats = [float(val) for val in angles]

    return np.array(angles_floats), np.array(powers_floats)
