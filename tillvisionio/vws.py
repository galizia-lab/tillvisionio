import textfsm
import pkg_resources
import pandas as pd
import numpy as np
import pathlib as pl
from scipy.spatial import cKDTree


def get_vws_log_template_filename():
    return pkg_resources.resource_filename('tillvisionio',
                                           "textfsm_templates/VWS_LOG_FSM_TEMPLATE.txt")


def load_pst(filename):
    """
    Implementation based on legacy code by Giovanni Galizia and Georg Raiser
    read tillvision based .pst files as uint16.
    """
    # filename can have an extension (e.g. .pst), or not
    # reading stack size from inf
    #inf_path = os.path.splitext(filename)[0] + '.inf'
    #this does not work for /data/030725bR.pst\\dbb10F, remove extension by hand,
    #assuming it is exactly 3 elements
    if filename[-4] == '.':
        filename = filename[:-4] #reomove extension
    meta = {}
    with open(filename+'.inf','r') as fh:
    #    fh.next()
        for line in fh.readlines():
            try:
                k,v = line.strip().split('=')
                meta[k] = v
            except:
                pass
    # reading stack from pst
    shape = np.int32((meta['Width'],meta['Height'],meta['Frames']))
    raw   = np.fromfile(filename+'.pst',dtype='int16')
    data  = np.reshape(raw,shape,order='F')

    # was swapping x, y axes; commented out to retain original order
    # data  = data.swapaxes(0,1)

    data = data.astype('uint16')
    return data


def get_pairs_closer_than(arr1, arr2, min_dists_arr_1):
    """
    For each element in arr1, finds the closest element in <arr2> whose distance is not more than the corresponding
    element in <min_dist_arr_1>. Returns an array of the same size as <arr1> with the index of the matching element of
    <arr2> if a match was found and "-1" otherwise.
    :param arr1: iterable of m-element-float-interables
    :param arr2: iterable of m-element-float-interables
    :param min_dists_arr_1: iterable of floats, of same size as <arr1>
    :return: iterable of int, same size as <arr1>
    """

    tree = cKDTree(np.array(arr1).reshape((len(arr1), 1)))
    arr2_2D = np.array(arr2).reshape((len(arr2), 1))
    dists, inds = tree.query(arr2_2D)

    pairs_too_far = [x > y for x, y in zip(dists, min_dists_arr_1)]

    inds[pairs_too_far] = -1

    return inds


class VWSDataManager(object):

    def __init__(self, vws_log_file):

        self.vws_log_file = vws_log_file
        self.vws_log_file_path = pl.Path(vws_log_file)

        with open(self.vws_log_file) as vws_fle:
            vws_log_text = vws_fle.read()

        with open(get_vws_log_template_filename()) as template_fle:
            textfsm_template = textfsm.TextFSM(template_fle)
            data = textfsm_template.ParseText(vws_log_text)
            self.data_df = pd.DataFrame(columns=textfsm_template.header, data=data)
            self.data_df.replace("", np.NaN, inplace=True)
            self.data_df = self.data_df.applymap(lambda x: pd.to_numeric(x, errors='ignore'))
            self.data_df.reset_index(drop=False, inplace=True)

    def get_earliest_utc(self):
        """
        Get the UTC of the first measurement
        :return: int
        """

        utcs = [x for x in self.data_df["UTCTime"].values if type(x) in (float, np.float64, int, np.int64)]
        return min(utcs)

    def get_all_metadata(self, filter=None, additional_cols_func=None):
        """
        Get all metadata entries
        :param filter: function with a series as input argument and returns True or False, indicating whether
        the measurement should be retained. The series will contain indices and values as defined in
        "VWS_LOG_FSM_TEMPLATE.txt"
        :param additional_cols_func: a function that takes a series as input and returns a dictionary whose key value
        pairs are used to add new columns to the returned dataframes. Note that if a key of the returned dictionary is
        an exisiting column, it will be overwritten.
        :return: pandas.DataFrame
        """

        if filter is None:
            filter = lambda s: True

        # filter measurements based on <filter>
        mask = self.data_df.apply(filter, axis=1)
        indices2use = mask.index.values[mask.values]
        data_df2return = self.data_df.reindex(index=indices2use)

        # add columns using <additional_cols_func>
        for ind, row in data_df2return.iterrows():
            if additional_cols_func is not None:
                additional_cols = additional_cols_func(row)
                for k, v in additional_cols.items():
                    data_df2return.loc[ind, k] = v

        return data_df2return

    def get_image_data(self, label):
        """
        Get the image data contained in the pst file corresponding to the entry with label <label>
        :param label: string
        :return: numpy.ndarray converted to uint16
        """

        subset_matching_label = self.data_df.loc[lambda s: s["Label"] == label, :]

        assert subset_matching_label.shape[0] == 1, f"More than one entries found in {self.vws_log_file}" \
                                                    f"with label={label}"

        dbb_file_name_as_recorded = subset_matching_label["Location"].iloc[0]

        dbb_file_path_as_recorded = pl.PureWindowsPath(dbb_file_name_as_recorded)

        dbb_file_path_now = self.vws_log_file_path.parent / dbb_file_path_as_recorded.parts[-2] / \
                            dbb_file_path_as_recorded.parts[-1]

        image_data = load_pst(str(dbb_file_path_now))

        return image_data

    def get_frame_posix_timestamps(self, frame_number, df_to_use=None):

        if df_to_use is None:
            df_to_use = self.data_df

        frame_timestamps_relative_to_start_ms = \
            df_to_use["Timing_ms"].apply(lambda s: float(s.split(" ")[frame_number])).values

        frame_timestamps = df_to_use["UTCTime"].values + frame_timestamps_relative_to_start_ms * 0.001

        return frame_timestamps

    def get_metadata_two_wavelengths(self, wavelengths, filter=None, additional_cols_func=None):
        """
        Assumes that measurements consist of simultaneous imaging with the two wavelengths specied. Finds all
        corresponding pairs and return them in two pandas dataframes with row correspondence between them.
        :param wavelengths: iterable of 2 floats, the wavelengths used in nm
        :param filter: function that takes a series as input and returns True or False, indicating whether
        the measurement should be retained. The series will contain indices and values as defined in
        "VWS_LOG_FSM_TEMPLATE.txt"
        :param additional_cols_func: a function that takes a series as input and returns a dictionary whose key value
        pairs are used to add new columns to the returned dataframes. Note that if a key of the returned dictionary is
        an exisiting column, it will be overwritten.
        :return: pandas.DataFrame, pandas.DataFrame
        """

        assert len(wavelengths) == 2, f"the parameter <wavelength> is expected to be a two member iterable of floats," \
                                      f" got {wavelengths}"

        all_df = self.get_all_metadata(filter, additional_cols_func)

        wavelength1_mask = all_df["MonochromatorWL_nm"].apply(lambda x: np.allclose(x, wavelengths[0]))
        wavelength2_mask = all_df["MonochromatorWL_nm"].apply(lambda x: np.allclose(x, wavelengths[1]))

        wavelength1_df = all_df.loc[wavelength1_mask, :]
        wavelength2_df = all_df.loc[wavelength2_mask, :]

        wavelength1_first_timestamps = self.get_frame_posix_timestamps(1, df_to_use=wavelength1_df)
        wavelength1_second_timestamps = self.get_frame_posix_timestamps(2, df_to_use=wavelength1_df)
        wavelength1_last_timestamps = self.get_frame_posix_timestamps(-1, df_to_use=wavelength1_df)

        wavelength2_first_timestamps = self.get_frame_posix_timestamps(1, df_to_use=wavelength2_df)
        wavelength2_second_timestamps = self.get_frame_posix_timestamps(2, df_to_use=wavelength2_df)
        wavelength2_last_timestamps = self.get_frame_posix_timestamps(-1, df_to_use=wavelength2_df)

        wavelength1_dts = np.array(wavelength1_second_timestamps) - np.array(wavelength1_first_timestamps)
        wavelength2_dts = np.array(wavelength2_second_timestamps) - np.array(wavelength2_first_timestamps)

        min_dists = [min(x, y) for x, y in zip(wavelength1_dts, wavelength2_dts)]

        wl2_first_inds = get_pairs_closer_than(wavelength1_first_timestamps, wavelength2_first_timestamps,
                                               min_dists)
        wl2_last_inds = get_pairs_closer_than(wavelength1_last_timestamps, wavelength2_last_timestamps,
                                              min_dists)

        wl1_result_df = pd.DataFrame()
        wl2_result_df = pd.DataFrame()

        for ind, (wl2_first_ind, wl2_last_ind) in enumerate(zip(wl2_first_inds, wl2_last_inds)):

            if (wl2_first_ind >= 0) and (wl2_last_ind >= 0) and (wl2_first_ind == wl2_last_ind):

                wl1_result_df = wl1_result_df.append(wavelength1_df.iloc[ind, :], ignore_index=True)
                wl2_result_df = wl2_result_df.append(wavelength2_df.iloc[wl2_last_ind, :], ignore_index=True)

        return wl1_result_df, wl2_result_df











