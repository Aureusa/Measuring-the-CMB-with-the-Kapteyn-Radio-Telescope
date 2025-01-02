import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Callable, Any

from utils import Uncertainties


T_C = 77.36  # K +- 0.01
T_H = 275  # K +- 0.5
T_ATM = 275  # K +- 0.5


class CMBEstimator(Uncertainties):
    """
    CMB and thao_0 esimator for data without a satelite.
    """

    def __init__(self, angles: np.ndarray, powers: np.ndarray) -> None:
        """
        A way of instantiating the estimator.

        :param angles: the angles
        :type angles: np.ndarray
        :param powers: the powers
        :type powers: np.ndarray
        """
        self._angles = angles
        self._powers = self._dBm_to_W(powers)

    def get_estimate(self, initial_guesses: list[float], slicing: int):
        """
        Gets an esimtae for the cmb and the thao_0.

        :param initial_guesses: the initial guess in the form:
        [cmb_temp, thao]
        :type initial_guesses: list[float]
        :param slicing: the slicing to be applied to the data to discard
        big angles. Recomended value = -5.
        :type slicing: int
        """
        angles, powers = self._preprocess_data()

        t_ant, t_ant_error = self._get_relevant_temperatures(powers)

        popt, pcov_diag = self._fit_model(
            self._antena_model, initial_guesses, angles[:slicing], t_ant[:slicing]
        )

        t_cmb = tuple((popt[0], pcov_diag[0]))
        thao_0 = tuple((popt[1], pcov_diag[1]))

        self._plot_fit(
            self._antena_model,
            t_ant,
            t_ant_error,
            popt,
            pcov_diag,
            slicing,
            fit=False,
            title=f"The antena temperature with the last {-slicing} points discarded.",
        )
        self._plot_fit(
            self._antena_model,
            t_ant,
            t_ant_error,
            popt,
            pcov_diag,
            slicing,
            fit=True,
            title="The antena temperature with the fitted antena model.",
        )

        self._plot_antena_temperature_(
            angles,
            t_ant,
            t_ant_error,
            title="The contribution of both the CMB and the atmosphere\nto the antena temperature",
            fit=True,
            model1=self._cmb_temp_model,
            popt1=popt,
            model2=self._atmosphere_temp_model,
            popt2=[popt[1]]
            )

        # Prints the CMB temperature and tau_0
        print(f"T_cmb = {popt[0]} \\pm {pcov_diag[0]}")
        print(f"tau_0 = {popt[1]} \\pm {pcov_diag[1]}")

        # Prints the slicing
        print(f"Applied slicing: {slicing}")

        return t_cmb, thao_0

    def _get_relevant_temperatures(
        self, powers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Calculates the relevent temperatures like
        T_rec (reciever temperature), T_sys (system temperature), and
        T_ant (antena temperature).

        :param powers: the powers
        :type powers: _type_
        :return: np.ndarray
        :rtype: tuple[np.ndarray, np.ndarray, float]
        """
        g_cal, g_cal_error = self._estimate_g_cal()

        t_rec, t_rec_error = self._estimate_t_rec()

        t_sys, t_sys_error = self._estimate_t_sys(g_cal, powers, g_cal_error)

        t_ant, t_ant_error = self._generate_t_ant_data(
            t_sys, t_rec, t_sys_error, t_rec_error
        )

        # Prints the calibration constant and the temperature of the reciever
        print(f"G_cal = {g_cal} \\pm {g_cal_error}")
        print(f"T_rec = {t_rec} \\pm {t_rec_error}")

        return t_ant, t_ant_error

    def plot_antena(self):
        """
        Plots the antenna temperature as a function of angles by preprocessing
        the data and retrieving the relevant temperature values.

        This function preprocesses the data using `_preprocess_data`,
        retrieves the relevant antenna temperatures and their errors using
        `_get_relevant_temperatures`, and then
        calls `_plot_antena_temperature_` to generate the plot.
        """
        angles, powers = self._preprocess_data()

        t_ant, t_ant_error = self._get_relevant_temperatures(powers)

        self._plot_antena_temperature_(angles, t_ant, t_ant_error)

    def _plot_antena_temperature_(
        self,
        angles,
        t_ant,
        t_ant_error,
        title: str = "Antena temperature in terms of angles",
        fit: bool = False,
        model1: Callable[..., Any] | None = None,
        popt1: list[float] | None = None,
        model2: Callable[..., Any] | None = None,
        popt2: list[float] | None = None,
        model3: Callable[..., Any] | None = None,
        popt3: list[float] | None = None,
    ):
        """
        Plots the antenna temperature as a function of angles,
        with optional error bars and model fits.

        :param angles: Array of angles (in degrees) at which the
        antenna temperature is measured.
        :type angles: np.ndarray
        :param t_ant: Measured antenna temperatures at the
        corresponding angles.
        :type t_ant: np.ndarray
        :param t_ant_error: Error values associated with the
        antenna temperature measurements.
        :type t_ant_error: np.ndarray
        :param title: The title of the plot
        (default is "Antena temperature in terms of angles").
        :type title: str, optional
        :param fit: If True, will add fitted models to the plot.
        :type fit: bool, optional
        :param model1: The first model function to fit and plot,
        or None if no model is provided.
        :type model1: Callable[..., Any] | None, optional
        :param popt1: The optimized parameters for the first model,
        or None if no model is provided.
        :type popt1: list[float] | None, optional
        :param model2: The second model function to fit and plot,
        or None if no model is provided.
        :type model2: Callable[..., Any] | None, optional
        :param popt2: The optimized parameters for the second model,
        or None if no model is provided.
        :type popt2: list[float] | None, optional
        :param model3: The third model function to fit and plot,
        or None if no model is provided.
        :type model3: Callable[..., Any] | None, optional
        :param popt3: The optimized parameters for the third model,
        or None if no model is provided.
        :type popt3: list[float] | None, optional
        """
        _, ax = plt.subplots()

        ax.errorbar(
            angles,
            t_ant,
            yerr=t_ant_error,
            capsize=5,
            ecolor="black",
            elinewidth=1,
            fmt="o",
            label="Antena temperature",
            color="blue",
        )

        if fit:
            self._add_fits_to_plot(
                ax, angles, model1, popt1, model2, popt2, model3, popt3
            )

        ax.set_xlabel("Angles (deg)")
        ax.set_ylabel("Temperature (K)")
        ax.set_title(title)
        ax.legend()

        plt.show()

    def _add_fits_to_plot(
        self,
        ax,
        angles: np.ndarray,
        model1: Callable[..., Any] | None = None,
        popt1: list[float] | None = None,
        model2: Callable[..., Any] | None = None,
        popt2: list[float] | None = None,
        model3: Callable[..., Any] | None = None,
        popt3: list[float] | None = None,
    ) -> None:
        """
        Adds fitted models to a plot based on the provided angle
        data and model parameters.

        :param ax: The matplotlib axis object to which the fitted
        models will be plotted.
        :type ax: matplotlib.axes.Axes
        :param angles: Array of angles at which the models are evaluated.
        :type angles: np.ndarray
        :param model1: The first model function to fit and plot,
        or None if no model is provided.
        :type model1: Callable[..., Any] | None
        :param popt1: The optimized parameters for the first model,
        or None if no model is provided.
        :type popt1: list[float] | None
        :param model2: The second model function to fit and plot,
        or None if no model is provided.
        :type model2: Callable[..., Any] | None
        :param popt2: The optimized parameters for the second model,
        or None if no model is provided.
        :type popt2: list[float] | None
        :param model3: The third model function to fit and plot,
        or None if no model is provided.
        :type model3: Callable[..., Any] | None
        :param popt3: The optimized parameters for the third model,
        or None if no model is provided.
        :type popt3: list[float] | None
        """
        angles = np.linspace(angles.min(), angles.max(), 500)
        if model1 is not None:
            model_y = model1(angles, *popt1)
            ax.plot(
                angles,
                model_y,
                label=f"CMB contribution\nT_cmb = {popt1[0]:.2f}",
                color="orange",
            )
        if model2 is not None:
            model_y = model2(angles, *popt2)
            ax.plot(
                angles,
                model_y,
                label="Atmoshpere Contribution",
                color="green",
            )
        if model3 is not None:
            model_y = model3(angles, *popt3)
            ax.plot(
                angles,
                model_y,
                label="Satellite Contribution",
                color="red",
            )

    def _plot_fit(
        self,
        model: Callable[..., Any],
        data_y: np.ndarray,
        data_y_errors: np.ndarray,
        popt: list[float],
        pcov_diag: list[float],
        slicing: int | None,
        fit: bool = True,
        title: str = "Data from Kapteyn Radio Telescope with the fitted model",
    ):
        """
        Plots the fitted model.

        :param model: The model we are trying to fit
        as a callable function
        :type model: Callable[..., Any]
        :param data_y: the data used to fit the model,
        in this case the T_sys (system temperature)
        :type data_y: np.ndarray
        :param data_y_errors: the error in the data used to fit the model,
        in this case the error of T_sys (system temperature)
        :type data_y_errors: np.ndarray
        :param popt: the fitted parameters
        :type popt: list[float]
        :param pcov_diag: the errors of the fitted parameters
        :type pcov_diag: list[float]
        :param slicing: the slicing used to discard data close to
        the horizon
        :type slicing: int | None
        :param fit: if True shows the Fit, defaults to True
        :type fit: bool
        :param title: the title of the plot
        :type title: str
        """
        _, ax = plt.subplots()

        data_x = self._angles[2:-2]

        model_x = data_x[:slicing]
        data_x = data_x[:slicing]
        data_y = data_y[:slicing]
        data_y_errors = data_y_errors[:slicing]

        if fit:
            model_y = model(model_x, *popt)
            ax.plot(
                model_x,
                model_y,
                label="Modeled data:\n"
                f"T_cmb = {round(popt[0],2)}\u00b1{round(pcov_diag[0],2)}  K"
                f"\n thao_0 = {round(popt[1],2)}\u00b1{round(pcov_diag[1]):.2f}",
                color="red",
            )
        ax.errorbar(
            data_x,
            data_y,
            yerr=data_y_errors,
            capsize=5,
            ecolor="black",
            elinewidth=1,
            fmt="o",
            label="Antena Temperature",
            color="blue",
        )
        ax.set_title(title)
        ax.set_xlabel("Andles (deg)")
        ax.set_ylabel("Temperature (K)")
        ax.legend()

        plt.show()

    @staticmethod
    def _dBm_to_W(power: np.ndarray) -> np.ndarray:
        """
        Converts power in dBm to Watts

        :param power: power in dBm
        :type power: np.ndarray
        :return: oower in Watt
        :rtype: np.ndarray
        """
        exponent = power * 10 ** (-1)
        power_mili_watt = 10 ** (exponent)
        power_watt = power_mili_watt * 0.001
        return power_watt

    def _hot_and_cold_power(self) -> tuple[float, float]:
        """
        Calculates the hot and cold power

        :return: the P_hot and P_cold as tuple
        :return type: tuple
        """
        p_min = (self._powers[0] + self._powers[-1]) / 2
        p_max = (self._powers[1] + self._powers[-2]) / 2
        return p_max, p_min

    def _estimate_g_cal(self) -> tuple[float, float]:
        """
        Estimates the G calibration.

        :return: the G calibration and its error
        :rtype: tuple[float,float]
        """
        p_max, p_min = self._hot_and_cold_power()
        g_cal = (p_max - p_min) / (T_H - T_C)
        g_cal_error_ = self._g_cal_error(T_H, T_C, p_max, p_min, 0.5, 0.01)
        return g_cal, g_cal_error_

    def _estimate_t_sys(
        self, g_cal: float, power_watt: np.ndarray, g_cal_error: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates the temperature of the system.

        :param g_cal: the G calibration
        :type g_cal: float
        :param power_watt: the power in watts after preprocessing
        :type power_watt: np.ndarray
        :return: the estsimate T sys at the different points and its errors
        :rtype: tuple[np.ndarray,np.ndarray]
        """
        t_sys = power_watt / g_cal
        t_sys_error_ = self._t_sys_error(power_watt, g_cal_error, g_cal)
        return t_sys, t_sys_error_

    def _estimate_t_rec(self) -> tuple[float, float]:
        """
        Estimates the temperature of the reciever.

        :return: the T_rec and its error
        :rtype: tuple[float, float]
        """
        p_max, p_min = self._hot_and_cold_power()
        y = p_max / p_min
        t_rec = (T_H - y * T_C) / (y - 1)
        t_rec_error_ = self._t_rec_error(y, 0.5, 0.01)
        return t_rec, t_rec_error_

    def _generate_t_ant_data(
        self, t_sys: np.ndarray, t_rec, t_sys_error: np.ndarray, t_rec_error: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates the data for the temperature of the antena.

        :param t_sys: the temperature of the system
        :type t_sys: np.ndarray
        :param t_rec: the temperature of the reciever
        :type t_rec: _type_
        :return: the data for the T_ant and the errors
        :rtype: tuple[np.ndarray,np.ndarray]
        """
        t_ant = t_sys - t_rec
        t_ant_error_ = self._t_ant_error(t_sys_error, t_rec_error)
        return t_ant, t_ant_error_

    def _preprocess_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get rid of the hot and cold load measurments of the data.

        :param data1: _description_
        :type data1: np.ndarray
        :param data2: _description_
        :type data2: np.ndarray
        :return: _description_
        :rtype: _type_
        """
        angles = self._angles[2:-2]
        powers = self._powers[2:-2]
        return angles, powers

    @staticmethod
    def _antena_model(angles: np.ndarray, t_cmb: float, thao: float) -> np.ndarray:
        """
        Defining the anteno model, given by:
        T_ant = T_cmb*exp(-thao_0*sec(z))+T_atm*(1-exp(-thao_0*sec(z))),
        where z is the zenith angle.

        :param angles: the angles
        :type angles: np.ndarray
        :param t_cmb: the cmb temperature
        :type t_cmb: float
        :type thao: float
        :return: the temeprature of the antena.
        :rtype: np.ndarray
        """
        zenith_angle = angles - 90
        z = np.radians(zenith_angle)
        t_ant = t_cmb * np.exp(-thao / np.cos(z)) + T_ATM * (
            1 - np.exp(-thao / np.cos(z))
        )
        return t_ant

    def _fit_model(
        self,
        model: Callable[..., Any],
        guesses: list[float],
        angles: np.ndarray,
        t_ant: np.ndarray,
        method="dogbox",
    ) -> tuple[list[float], list[float]]:
        """
        This function is used to fit the model using initial guesses,
        the angles of observation and the antena temperature at each
        angle.

        :param model: the model we are trying to fit.
        :type model: Callable[..., Any]
        :param guesses: the initial guesses
        :type guesses: list[float]
        :param angles: the angles
        :type angles: np.ndarray
        :param t_ant: the antena temperature
        :type t_ant: np.ndarray
        :param method: the method used to performe the curve_fit,
        defaults to "dogbox"
        :type method: str, optional
        :return: the fitted parameters as well as their respective
        uncertainties (errors)
        :rtype: tuple[list[float],list[float]]
        """
        popt, pcov = curve_fit(
            model, xdata=angles, ydata=t_ant, p0=guesses, method=method
        )
        return popt, np.diag(pcov).tolist()

    def _cmb_temp_model(
        self, angles: np.ndarray, t_cmb: float, thao: float
    ) -> np.ndarray:
        """
        Models the Cosmic Microwave Background (CMB) temperature
        based on the zenith angle and atmospheric conditions.

        :param angles: Array of angles (in degrees) from the zenith direction.
        :type angles: np.ndarray
        :param t_cmb: Temperature of the Cosmic Microwave Background (CMB)
        at the zenith angle.
        :type t_cmb: float
        :param thao: Atmospheric opacity or thickness, affecting the
        temperature distribution.
        :type thao: float
        :return: Modeled temperature of the CMB at the given angles.
        :rtype: np.ndarray
        """
        zenith_angle = angles - 90
        z = np.radians(zenith_angle)
        t_cmb_modeled = t_cmb * np.exp(-thao / np.cos(z))
        return t_cmb_modeled

    def _atmosphere_temp_model(self, angles: np.ndarray, thao: float) -> np.ndarray:
        """
        Models the atmospheric temperature based on the zenith
        angle and atmospheric opacity.

        :param angles: Array of angles (in degrees) from the zenith direction.
        :type angles: np.ndarray
        :param thao: Atmospheric opacity or thickness, affecting the
        temperature distribution.
        :type thao: float
        :return: Modeled temperature of the atmosphere at the given angles.
        :rtype: np.ndarray
        """
        zenith_angle = angles - 90
        z = np.radians(zenith_angle)
        t_atm_modeled = T_ATM * (1 - np.exp(-thao / np.cos(z)))
        return t_atm_modeled


class CMBEstimatorWithSatelite(CMBEstimator):
    """
    CMB and thao_0 esimator for data with a satelite.
    """

    def get_estimate(
        self,
        initial_guesses_antena: list[float],
        initial_guesses_gaussian: list[float],
        slicing: int,
    ):
        """
        Gets an esimtae for the cmb and the thao_0.

        :param initial_guesses: the initial guess in the form:
        [cmb_temp, thao]
        :type initial_guesses: list[float]
        :param initial_guesses_gaussian: the initial guess in the form:
        [amp, mu, sigma, offset]
        :type initial_guesses_gaussian: list[float]
        :param slicing: the slicing to be applied to the data to discard
        big angles. Recomended value = -5.
        :type slicing: int
        """
        # Preprocess the data
        angles, powers = self._preprocess_data()

        # Get the relevant temperatures
        t_ant, t_ant_error = self._get_relevant_temperatures(powers)

        # Fit a gaussian to the antena temperature in order to model the satelite
        popt_gauss, _ = self._gaussian_fit(
            initial_guesses_gaussian, angles, t_ant, t_ant, t_ant_error
        )

        # Remove the contributions from the satelite from the initial data
        modeled_values = self._gaussian_model(angles, *popt_gauss)
        t_ant_decreased = t_ant - modeled_values

        # Plot the antena after satelite removal
        self._plot_antena_temperature_(
            angles,
            t_ant_decreased,
            t_ant_error,
            title="Antena temperature after removing the contribution from the satellite",
        )

        # Fit the antena model to the decreased antena temperature
        popt, pcov_diag = self._antena_fit(
            initial_guesses_antena, angles, t_ant_decreased, t_ant_error, slicing - 1
        )

        # Plot the initial antena temperature with the different contributions
        self._plot_antena_temperature_(
            angles,
            t_ant,
            t_ant_error,
            title="The contribution from the CMB, the atmosphere, and the satellite\nto the antena temperature",
            fit=True,
            model1=self._cmb_temp_model,
            popt1=popt,
            model2=self._atmosphere_temp_model,
            popt2=[popt[1]],
            model3=self._gaussian_model,
            popt3=popt_gauss,
        )

        # Get the final results for the T_cmb and thao_0 with their uncertainties
        t_cmb = tuple((popt[0], pcov_diag[0]))
        thao_0 = tuple((popt[1], pcov_diag[1]))

        # Print the CMB temperature an tau_0
        print(f"T_cmb = {popt[0]} \\pm {pcov_diag[0]}")
        print(f"tau_0 = {popt[1]} \\pm {pcov_diag[1]}")

        # Prints the slicing
        print(f"Applied slicing: {slicing}")

        return t_cmb, thao_0

    def _antena_fit(
        self,
        initial_guesses_antena: list[float],
        angles: np.ndarray,
        t_ant: np.ndarray,
        t_ant_error: np.ndarray,
        slicing: int,
    ) -> list[float]:
        """
        Makes an initial antena fit by removing the ignoring the data points
        suspected to be influenced by the satelite.

        :param initial_guesses_antena: initial guesses for the anteana model
        :type initial_guesses_antena: list[float]
        :param angles: the angles
        :type angles: np.ndarray
        :param t_ant: the antena temperatures
        :type t_ant: np.ndarray
        :param t_ant_error: the error in the antena temperatures
        :type t_ant_error: np.ndarray
        :param slicing: the slicing of the data
        :type slicing: int
        :param title: the title of the plot
        :type title: str
        :return: the result from the fit
        :rtype: list[float]
        """
        # Fit the antena model
        popt, pcov_diag = self._fit_model(
            self._antena_model,
            initial_guesses_antena,
            angles[:slicing],
            t_ant[:slicing],
        )

        # Plot the sliced antena temperatures
        self._plot_fit(
            self._antena_model,
            t_ant,
            t_ant_error,
            popt,
            pcov_diag,
            slicing,
            fit=False,
            title=f"The antena temperature with the last {-slicing} points discarded.",
        )

        # Plot the antena fit
        self._plot_fit(
            self._antena_model,
            t_ant,
            t_ant_error,
            popt,
            pcov_diag,
            slicing,
            fit=True,
            title="The antena temperature with the fitted antena model.",
        )

        return popt, pcov_diag

    def _gaussian_fit(
        self,
        initial_guesses_gaussian: list[float],
        angles: np.ndarray,
        t_ant_decreased: np.ndarray,
        t_ant: np.ndarray,
        t_ant_error: np.ndarray,
    ) -> list[float]:
        """
        Perform the gaussian fit used to model the satelite contribution
        in the data using the antena temperature without the contribution
        by the CMB.

        :param initial_guesses_gaussian: the initial guesses for the gaussian
        parameters
        :type initial_guesses_gaussian: list[float]
        :param angles: the angles
        :type angles: np.ndarray
        :param t_ant_decreased: the antena temperature without the CMB contribution
        :type t_ant_decreased: np.ndarray
        :param t_ant: the antena temperature
        :type t_ant: np.ndarray
        :param t_ant_error: the error in the antena temperature
        :type t_ant_error: np.ndarray
        :return: the fitted gaussian parameters
        :rtype: list[float]
        """
        # Remove the last point from the data for a better fit
        t_ant_decreased = t_ant_decreased[:-1]
        angles = angles[:-1]
        t_ant_error = t_ant_error[:-1]

        # Scale the antena signal down
        t_ant_scaled = t_ant_decreased / np.max(t_ant_decreased)

        # Fit a gaussian to the down scaled signal
        popt_gauss, pcov_gauss_diag = self._fit_model(
            self._gaussian_model,
            initial_guesses_gaussian,
            angles,
            t_ant_scaled,
            method="trf",
        )

        # Upscale the gaussian fit
        popt_gauss[0] *= np.max(t_ant_decreased)

        # Plot the gaussian fit
        self._plot_gaussian_fit(angles, t_ant_decreased, t_ant_error, popt_gauss)

        return popt_gauss, pcov_gauss_diag

    def _plot_gaussian_fit(
        self, data_x, data_y: np.ndarray, data_y_errors: np.ndarray, popt: list[float]
    ):
        """
        Used to plot tha gaussian fit that models the sateilte
        in the data.

        :param data_y: the antena temperature scaled down
        :type data_y: np.ndarray
        :param popt: the fitted parameters of the gaussian fit
        :type popt: list[float]
        """
        _, ax = plt.subplots()

        x_gaussian = np.linspace(data_x.min(), data_x.max(), 1000)
        y_gaussian = self._gaussian_model(x_gaussian, *popt)

        ax.plot(x_gaussian, y_gaussian, label="Gaussian fit", color="red")
        ax.errorbar(
            data_x,
            data_y,
            yerr=data_y_errors,
            capsize=5,
            ecolor="black",
            elinewidth=1,
            fmt="o",
            label="Antena Temperature ",
            color="blue",
        )
        ax.set_title("Antena temperature with the fitted Gaussian model")
        ax.set_xlabel("Angles (deg)")
        ax.set_ylabel("Temperature (K)")
        ax.legend()

        plt.show()

    @staticmethod
    def _gaussian_model(
        angles: np.ndarray, amp: float, mu: float, sigma: float, offset: float
    ) -> np.ndarray:
        """
        Defining a simple gaussian model to model the satelite transit.

        :param angles: the angles
        :type angles: np.ndarray
        :param amp: the amplitude of the gaussian
        :type amp: float
        :param mu: the mean of the gaussian
        :type mu: float
        :param sigma: the standart deviation of tha gaussian
        :type sigma: float
        :param offset: the offset
        :type offset: float
        :return: the gaussian
        :rtype: np.ndarray
        """
        zenith_angle = np.radians(angles - 90)
        gaussian = np.exp(-((zenith_angle - mu) ** 2) / (2 * sigma**2))
        return amp * gaussian + offset
