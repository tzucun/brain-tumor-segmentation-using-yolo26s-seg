import unittest

import pandas as pd
import torch

from tune_thresholds import ThresholdValidator, operating_metrics, rank_common_thresholds


class ThresholdTests(unittest.TestCase):
    def test_ground_truth_is_rematched_after_confidence_filter(self):
        validator = ThresholdValidator(confidence_grid=[.1, .5], args={"plots": False})
        target = {"cls": torch.tensor([0]), "bboxes": torch.tensor([[0., 0., 10., 10.]]),
                  "masks": torch.ones((1, 4, 4))}
        masks = torch.ones((2, 4, 4))
        masks[0, 0, :] = 0
        pred = {"cls": torch.tensor([0., 0.]), "conf": torch.tensor([.9, .2]),
                "bboxes": torch.tensor([[0., 0., 9., 9.], [0., 0., 10., 10.]]), "masks": masks}
        validator.collect_operating_points(pred, target)
        rows, _ = validator.operating_rows()
        self.assertEqual(rows[0]["mask_tp"], 1)
        self.assertEqual(rows[0]["mask_fp"], 1)
        self.assertEqual(rows[1]["mask_tp"], 1)
        self.assertEqual(rows[1]["mask_fp"], 0)
        self.assertEqual(rows[1]["box_tp"], 1)

    def test_negative_false_positive_rate_is_per_image(self):
        validator = ThresholdValidator(confidence_grid=[.25], args={"plots": False})
        target = {"cls": torch.empty(0), "bboxes": torch.empty((0, 4)), "masks": torch.empty((0, 4, 4))}
        pred = {"cls": torch.tensor([0., 1.]), "conf": torch.tensor([.8, .9]),
                "bboxes": torch.zeros((2, 4)), "masks": torch.ones((2, 4, 4))}
        validator.collect_operating_points(pred, target)
        empty = {key: value[:0] for key, value in pred.items()}
        validator.collect_operating_points(empty, target)
        rows, _ = validator.operating_rows()
        self.assertEqual(rows[0]["negative_images"], 2)
        self.assertEqual(rows[0]["false_positive_images"], 1)
        self.assertEqual(rows[0]["negative_fp_image_rate"], .5)
        self.assertEqual(rows[0]["false_detections_on_negatives"], 2)

    def test_no_ground_truth_does_not_make_recall_perfect(self):
        self.assertIsNone(operating_metrics(0, 0, 0)["recall_micro"])
        self.assertIsNone(operating_metrics(0, 0, 0)["f1_micro"])

    def test_ranking_rejects_incomplete_model_pair(self):
        row = {"model_id": "tanpa_negative", "mode": "nms", "confidence": .25, "iou_nms": .7,
               "mask_f1_micro": 1., "mask_recall_micro": 1., "negative_fp_image_rate": 0.}
        self.assertTrue(rank_common_thresholds(pd.DataFrame([row])).empty)


if __name__ == "__main__":
    unittest.main()
