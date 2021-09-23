from tillvisionio.vws import VWSDataManager
import pandas as pd
import pathlib as pl


def VWS_read_test():
    """
    Testing the initialization of a tillvisionio.vws.VWSDataManager with a vws.log file
    """
    test_file = "tests/testFiles/HS_bee_OXON_PELM_180718.vws.log"
    expected_xl = "tests/testFiles/HS_bee_OXON_PELM_180718.vws.xls"
    vws_data = VWSDataManager(test_file)
    expected_df = pd.read_excel(expected_xl, index_col=0)
    read_df = vws_data.get_all_metadata()

    temp1 = read_df.applymap(lambda x: pd.to_numeric(x, errors="ignore", downcast="float"))
    temp2 = expected_df.applymap(lambda x: pd.to_numeric(x, errors="ignore", downcast="float"))
    assert temp1.equals(temp2)


# def vws_segregation_test():
#     """
#     Testing the functions get_snapshop_metadata and get_measurement_metadata of tillvisionio.vws.VWSDataManager
#     """
#
#     test_file = "tests/testFiles/HS_bee_OXON_PELM_180718.vws.log"
#     expected_snapshots_only_xl = "tests/testFiles/HS_bee_OXON_PELM_180718_snapshots_only.vws.xls"
#     expected_measurements_only_xl = "tests/testFiles/HS_bee_OXON_PELM_180718_measurements_only.vws.xls"
#
#     expected_snapshots_only_df = pd.read_excel(expected_snapshots_only_xl)
#     expected_snapshots_only_df = expected_snapshots_only_df.apply(pd.to_numeric, downcast="float", errors="ignore")
#     expected_measurements_only_df = pd.read_excel(expected_measurements_only_xl)
#     expected_measurements_only_df = expected_measurements_only_df.apply(pd.to_numeric, downcast="float",
#                                                                         errors="ignore")
#
#     vws_data = VWSDataManager(test_file)
#     snapshots_only_df = vws_data.get_snapshot_metadata()
#     snapshots_only_df.reset_index(drop=True, inplace=True)
#     snapshots_only_df = snapshots_only_df.apply(pd.to_numeric, errors="ignore", downcast="float")
#
#     measurements_only_df = vws_data.get_measurement_metadata()
#     measurements_only_df.reset_index(drop=True, inplace=True)
#     measurements_only_df = measurements_only_df.apply(pd.to_numeric, errors="ignore", downcast="float")
#
#     assert measurements_only_df.equals(expected_measurements_only_df)
#     assert snapshots_only_df.equals(expected_snapshots_only_df)


def load_image_data_test():
    """
    Testing the function get_image_data of tillvisionio.vws.VWSDataManager
    """

    test_file = "tests/testFiles/HS_bee_OXON_PELM_180718.vws.log"

    labels = ["syringe00_NONL", "00_MOL", "01_LINT-4"]

    vws = VWSDataManager(test_file)

    for label in labels:

        img_data = vws.get_image_data(label)



