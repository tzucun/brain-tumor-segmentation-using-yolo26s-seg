import unittest

import numpy as np
from PIL import Image

from comparison import class_prediction_frame, experiment_key, mask_difference
from inference import CLASS_NAMES, MODEL_SPECS, PredictionOutput


def output(masks=None, detections=None):
    masks = masks or {name: np.zeros((4, 4), dtype=bool) for name in CLASS_NAMES}
    detections = detections or []
    return PredictionOutput(Image.new("RGB", (4, 4)), detections, 0.0, 0.0, "", "-", "-", masks)


class ComparisonTests(unittest.TestCase):
    def test_shared_area_with_different_class_is_not_agreement(self):
        left, right = output(), output()
        left.class_masks["glioma"][1, 1] = True
        right.class_masks["meningioma"][1, 1] = True
        _, counts = mask_difference(Image.new("RGB", (4, 4)), left, right)
        self.assertEqual(counts["Keduanya, kelas sama"], 0)
        self.assertEqual(counts["Keduanya, kelas berbeda"], 1)

    def test_empty_masks_are_not_an_accuracy_score(self):
        _, counts = mask_difference(Image.new("RGB", (4, 4)), output(), output())
        self.assertEqual(sum(counts.values()), 0)
        frame = class_prediction_frame({MODEL_SPECS[0].id: output()})
        self.assertTrue(frame["Confidence (%)"].isna().all())
        self.assertTrue((frame["Jumlah deteksi"] == 0).all())

    def test_additional_overlapping_class_is_marked_as_disagreement(self):
        left, right = output(), output()
        left.class_masks["glioma"][1, 1] = True
        left.class_masks["meningioma"][1, 1] = True
        right.class_masks["glioma"][1, 1] = True
        _, counts = mask_difference(Image.new("RGB", (4, 4)), left, right)
        self.assertEqual(counts["Keduanya, kelas sama"], 0)
        self.assertEqual(counts["Keduanya, kelas berbeda"], 1)

    def test_disabled_iou_does_not_change_end_to_end_key(self):
        self.assertEqual(experiment_key(b"image", "end2end", .25, .30),
                         experiment_key(b"image", "end2end", .25, .90))
        self.assertNotEqual(experiment_key(b"image", "nms", .25, .30),
                            experiment_key(b"image", "nms", .25, .90))

    def test_region_partition_counts_only_predicted_pixels(self):
        left, right = output(), output()
        left.class_masks["glioma"][0, 0] = True
        left.class_masks["glioma"][0, 1] = True
        right.class_masks["glioma"][0, 1] = True
        right.class_masks["glioma"][0, 2] = True
        _, counts = mask_difference(Image.new("RGB", (4, 4)), left, right)
        self.assertEqual(list(counts.values()), [1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
