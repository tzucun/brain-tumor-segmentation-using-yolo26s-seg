import unittest

import numpy as np
import pandas as pd
from PIL import Image

from charts import grouped_bars
from comparison import class_prediction_frame
from inference import CLASS_NAMES, MODEL_SPECS, PredictionOutput


class PredictionChartTests(unittest.TestCase):
    def make_output(self, masks_available=True):
        masks = {name: np.zeros((4, 4), dtype=bool) for name in CLASS_NAMES}
        masks["glioma"][0, 0] = True
        return PredictionOutput(
            Image.new("RGB", (4, 4)),
            [{"kelas": "glioma", "confidence": 0.8}, {"kelas": "glioma", "confidence": 0.9}],
            0.1, 6.25, "ok", "glioma", "glioma", masks, masks_available,
        )

    def test_scores_are_maximum_per_class_and_masks_are_union_area(self):
        frame = class_prediction_frame({MODEL_SPECS[0].id: self.make_output()})
        glioma = frame[frame["Kelas"] == "glioma"].iloc[0]
        self.assertEqual(glioma["Confidence (%)"], 90)
        self.assertEqual(glioma["Luas mask (%)"], 6.25)
        missing = frame[frame["Kelas"] == "meningioma"].iloc[0]
        self.assertTrue(pd.isna(missing["Confidence (%)"]))
        self.assertEqual(missing["Luas mask (%)"], 0)
        self.assertEqual(frame["Model"].nunique(), 1)

    def test_unavailable_masks_are_not_filled_with_zero(self):
        frame = class_prediction_frame({MODEL_SPECS[0].id: self.make_output(False)})
        self.assertTrue(frame["Luas mask (%)"].isna().all())

    def test_chart_has_labels_and_consistent_model_order(self):
        frame = class_prediction_frame({s.id: self.make_output() for s in MODEL_SPECS})
        order = [MODEL_SPECS[1].label, MODEL_SPECS[0].label]
        for dark in (False, True):
            chart = grouped_bars(frame, "Confidence (%)", dark, model_order=order).to_dict()
            self.assertEqual(chart["layer"][0]["encoding"]["color"]["scale"]["domain"], order)
            self.assertEqual(chart["layer"][1]["mark"]["type"], "text")
            self.assertEqual(chart["layer"][0]["encoding"]["y"]["scale"]["domain"], [0, 100])


if __name__ == "__main__":
    unittest.main()
