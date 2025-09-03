import numpy as np

from .builder import DATASETS
from .defaults import DefaultDataset


def segment2onehot(segment, num_classes):
    mask = segment != -1
    segment_onehot = np.zeros((len(segment), num_classes), dtype=np.float32)
    segment_onehot[mask, segment[mask]] = 1
    return segment_onehot


@DATASETS.register_module()
class WeldingDataset(DefaultDataset):
    VALID_ASSETS = [
        "coord",
        "color",
        "normal",
        "segment",
        "obj_segment",
        "obj_segment_onehot",
        "edge",
        "edge_ds",
        "visible_edge",
        "spl_t",
        "spl_c",
        "spl_k",
    ]

    def __init__(self, spline_coef_keys=None, batch_keys=None, **kwargs):
        self.spline_coef_keys = spline_coef_keys
        self.batch_keys = batch_keys
        super().__init__(**kwargs)

    def get_data(self, idx):
        data_dict = super().get_data(idx)

        if "obj_segment_onehot" not in data_dict:
            data_dict["obj_segment_onehot"] = segment2onehot(data_dict["obj_segment"], 2)

        if self.spline_coef_keys is not None:
            data_dict["spl_coef"] = np.concatenate([data_dict[key] for key in self.spline_coef_keys])

        if self.batch_keys is not None:
            for key in self.batch_keys:
                data_dict[key] = data_dict[key][None]

        data_dict["index_valid_keys"] = [
            "coord",
            "color",
            "normal",
            "segment",
            "obj_segment",
            "obj_segment_onehot",
        ]

        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict