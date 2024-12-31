from utils import unpack_data
from estimators import CMBEstimator, CMBEstimatorWithSatelite


def main():
    """
    Runs the main function for estimating the CMB temperature
    and thao_0 using the data without satelite.
    """
    print("==== Estimating from dataset without satelite ====")

    angles, powers = unpack_data("5.data155637_cmb.txt")

    estimator = CMBEstimator(angles, powers)

    estimator.get_estimate([3, 0.1], -5)


def main_satelite():
    """
    Runs the main function for estimating the CMB temperature
    and thao_0 using the data with satelite.
    """
    print("==== Estimating from dataset with satelite ====")

    angles, powers = unpack_data("12.data160408_cmb.txt")

    estimator = CMBEstimatorWithSatelite(angles, powers)

    estimator.get_estimate([3, 0.1], [1.0, 65, 20, 0.0], -12)


if __name__ == "__main__":
    main()
    main_satelite()
