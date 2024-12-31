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


class Uncertainties:
    """
    Calculates the relevent uncertainties needed for the analysis.
    """

    def _t_rec_error(self, y: float, T_H_error: float, T_C_error: float) -> float:
        """
        Returns the error in the reciever temperature

        :param y: the y value
        :type y: float
        :param t_h_error: the error in the hot load temperature
        :type t_h_error: float
        :param t_c_error: the error in the cold load temperature
        :type t_c_error: float
        :return: the error in the reciever temperature
        :rtype: float
        """
        term = 1 / (y - 1)
        return ((term * T_H_error) ** 2 + (term * T_C_error) ** 2) ** 0.5

    def _g_cal_error(self, T_H, T_C, P_H, P_C, sigma_T_H, sigma_T_C) -> float:
        """
        Calculates the error in the calibration constant

        :param T_H: hot load temperature
        :type T_H: float
        :param T_C: cold load temperature
        :type T_C: float
        :param P_H: power at the hot load
        :type P_H: float
        :param P_C: power at the cold load
        :type P_C: float
        :param sigma_T_H: Uncertainty in hot load temperature
        :type sigma_T_H: float
        :param sigma_T_C: Uncertainty in cold load temperature
        :type sigma_T_C: float
        :return: The uncertainty in the calibration constant
        :rtype: float
        """
        delta_T = T_H - T_C
        delta_P = P_H - P_C

        term1 = ((-delta_P) / (delta_T**2) * sigma_T_H) ** 2
        term2 = ((delta_P) / (delta_T**2) * sigma_T_C) ** 2

        sigma_G_cal = (term1 + term2) ** 0.5
        return sigma_G_cal

    def _t_sys_error(
        self, power: np.ndarray, sigma_G_cal: float, G_cal: float
    ) -> float:
        """
        Calculates the error in the system temperature

        :param power: measured powers
        :type power: np.ndarray
        :param sigma_G_cal: the uncertainty in G_cal
        :type sigma_G_cal: float
        :param G_cal: G_cal
        :type G_cal: float
        :return: the error in the system temperature
        :rtype: float
        """
        return power * sigma_G_cal / G_cal**2

    def _t_ant_error(self, sigma_T_sys: float, sigma_T_rec: float) -> float:
        """
        Calculates the error in the antena temperature.

        :param sigma_T_sys: Uncertainty in the system temperature
        :type sigma_T_sys: float
        :param sigma_T_rec: Uncertainty in the reciever temperature
        :type sigma_T_rec: float
        :return: the uncertainty in the antena temperature
        :rtype: float
        """
        return (sigma_T_sys**2 + sigma_T_rec**2) ** 0.5
