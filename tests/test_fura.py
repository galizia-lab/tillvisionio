from tillvisionio.vws import VWSDataManager


def test_fura_metadata_load():
    """
    Testing the function tillvisionio.vws.VWSDataManager.get_metadata_two_wavelengths

    """

    test_vws_log = "tests/testFiles/fura/190112_locust_ip.vws.log"

    vws_manager = VWSDataManager(test_vws_log)

    wl1_df, wl2_df = vws_manager.get_metadata_two_wavelengths(wavelengths=(340, 380))

    wl1_labels = ["Fluo340nm_00", "Fluo340nm_01"]
    wl2_labels = ["Fluo380nm_00", "Fluo380nm_01"]

    wl1_label_diff = set(wl1_df["Label"]).symmetric_difference(set(wl1_labels))
    assert wl1_label_diff == set()

    wl2_label_diff = set(wl2_df["Label"]).symmetric_difference(set(wl2_labels))
    assert wl2_label_diff == set()


if __name__ == "__main__":

    test_fura_metadata_load()

