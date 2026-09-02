# -*- coding: utf-8 -*-
"""วัดระยะห่างของโดเมนระหว่างชุดเทรนกับชุดที่เอาไปใช้จริง"""
import cv2
import numpy as np

from conftest import make_ekg
from ekg_rpeak import domain as dm
from ekg_rpeak.config import Config


def test_features_describe_a_known_pattern():
    m = np.zeros((40, 40), bool)
    m[:, 10:12] = True                      # เส้นตั้งหนา 2 px สูงเต็มภาพ
    f = dm.features(m)
    assert f['w_p90'] == 2 and f['h_med'] == 40
    assert abs(f['ink'] - 2 / 40) < 1e-6


def test_features_of_dotted_noise_show_no_vertical_continuity():
    m = np.zeros((40, 40), bool)
    m[::4, ::4] = True                      # จุดประแบบกริดที่รอด binarization
    assert dm.features(m)['h_med'] == 1


def test_longest_run_picks_the_biggest_block():
    flags = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1], bool)
    assert dm.longest_run(flags) == (4, 3, 7)


def test_longest_run_on_all_false():
    assert dm.longest_run(np.zeros(5, bool)) == (0, 0, 0)


def test_model_view_is_downscaled_to_imgsz():
    img, _, _ = make_ekg(w=3000)
    view = dm.model_view(img, Config())
    assert max(view.shape) == Config().crop_imgsz and view.dtype == bool


def test_distance_is_zero_for_identical_and_grows_apart():
    a = {'ink': 0.3, 'w_p90': 2.0, 'h_med': 3.0, 'h_p90': 9.0}
    b = dict(a)
    assert dm.distance(a, b) == 0.0
    b['ink'] = 0.15
    assert dm.distance(a, b) > 0.5


def test_distance_is_none_without_a_reference():
    assert dm.distance({'ink': 1, 'w_p90': 1, 'h_med': 1, 'h_p90': 1}, {}) is None


def test_train_features_ignores_a_directory_without_mosaics(tmp_path):
    assert dm.train_features(str(tmp_path)) == []


def test_image_features_reads_a_folder(tmp_path):
    img, _, _ = make_ekg(w=600)
    cv2.imwrite(str(tmp_path / 'a.png'), img)
    rows = dm.image_features(str(tmp_path), Config())
    assert len(rows) == 1 and set(dm.KEYS) <= set(rows[0])
    assert dm.summarize(rows)['ink'] > 0
