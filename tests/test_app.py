from __future__ import annotations

import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

import app
from inference import predict_image


class AppTests(unittest.TestCase):
    def test_initial_page_is_simple(self):
        page = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        self.assertEqual(len(page.exception), 0)
        self.assertEqual([slider.label for slider in page.slider], ["Confidence threshold"])
        self.assertEqual(len(page.tabs), 0)
        self.assertEqual(len(page.selectbox), 0)
        self.assertEqual(len(page.radio), 0)
        self.assertEqual(page.button[0].label, "Jalankan prediksi")
        self.assertTrue(page.button[0].disabled)
        self.assertFalse(page.expander[0].proto.expanded)

    def test_model_display_order(self):
        self.assertEqual([spec.id for spec in app.APP_MODELS],
                         ["dengan_negative", "tanpa_negative"])
        self.assertNotEqual(app.APP_MODELS[0].weights, app.APP_MODELS[1].weights)

    def test_prediction_details_are_collapsed_below_images(self):
        page = AppTest.from_string("""
from PIL import Image
import app
app.render_prediction({
    "image": Image.new("RGB", (8, 8)), "outputs": {}, "errors": {}, "file_name": "test.png",
})
""").run()
        self.assertEqual(len(page.exception), 0)
        self.assertEqual(page.expander[0].label, "Detail Hasil Prediksi dan Visualisasi")
        self.assertFalse(page.expander[0].proto.expanded)

    def test_viewer_is_native_and_escapes_title(self):
        with patch.object(app.st, "html") as render:
            app.render_prediction_image(Image.new("RGB", (8, 8)), '<img onerror="bad">', "viewer-test")
        markup = render.call_args.args[0]
        self.assertIn('popovertarget="viewer-test"', markup)
        self.assertIn('popover="auto"', markup)
        self.assertIn('popovertargetaction="hide"', markup)
        self.assertIn("data:image/png;base64,", markup)
        self.assertIn("&lt;img onerror=&quot;bad&quot;&gt;", markup)
        self.assertNotIn("<script", markup)

    def test_empty_outputs_do_not_draw_charts(self):
        with patch.object(app.st, "altair_chart") as chart:
            app.render_prediction_charts({}, False)
        chart.assert_not_called()

    def test_table_accepts_array_and_escapes_markup(self):
        with patch.object(app.st, "markdown") as markdown:
            app.render_html_table(app.pd.DataFrame({
                "bbox": [np.array([1, 2, 3, 4])], "<name>": ["<script>"],
            }))
        rendered = markdown.call_args.args[0]
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("&lt;name&gt;", rendered)

    def test_one_failed_model_does_not_stop_the_other(self):
        progress = Mock()
        with patch.object(app.st, "progress", return_value=progress), \
                patch.object(app, "model_revision", return_value="revision"), \
                patch.object(app, "load_model", side_effect=[RuntimeError("missing"), object()]), \
                patch.object(app, "get_device", return_value="cpu"), \
                patch.object(app, "predict_image", return_value="output"):
            outputs, errors = app.predict_both(Image.new("RGB", (8, 8)), 0.25)
        self.assertIn("dengan_negative", errors)
        self.assertEqual(outputs, {"tanpa_negative": "output"})
        progress.empty.assert_called_once()

    def test_end_to_end_does_not_pass_iou_to_predictor(self):
        model = SimpleNamespace(
            _app_inference_mode="end2end", _app_prediction_lock=threading.RLock(),
            predictor=SimpleNamespace(model=SimpleNamespace(end2end=True)),
            predict=Mock(return_value=[object()]),
        )
        with patch("inference.annotate_result", return_value=SimpleNamespace()) as annotate:
            predict_image(model, Image.new("RGB", (8, 8)), 0.25, None, "cpu")
        self.assertNotIn("iou", model.predict.call_args.kwargs)
        self.assertTrue(model.predict.call_args.kwargs["end2end"])
        annotate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
