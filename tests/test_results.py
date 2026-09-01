# -*- coding: utf-8 -*-
"""ผลที่เก็บลงดิสก์ต้องรอดการรีสตาร์ต และต้องไม่โกหกเมื่อของต้นทางเปลี่ยน"""
import json
import os

import numpy as np
import pytest

from ekg_rpeak import results as rs
from ekg_rpeak.config import Config


def fake_result():
    """โครงเดียวกับที่ pipeline คืน รวมชนิดจาก numpy ที่ json เขียนตรง ๆ ไม่ได้"""
    return {
        'raw': np.zeros((40, 60, 3), np.uint8),
        'boxes': np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=int),
        'rows': [[np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]],
        'peaks': [{'row': 0, 'index': 0, 'x': np.float64(1.5), 'y': 2.0,
                   'conf': np.float32(0.3), 'src': 'model', 'cls': 5, 'x_mm': 0.5}],
        'landmarks': [{'row': 0, 'x': 1.0, 'y': 2.0, 'conf': 0.1, 'cls': 3}],
        'grid': {'spacing': 40.0, 'lines': np.array([1.0, 41.0]), 'px_per_mm': 8.0,
                 'resid_rms_px': 0.1, 'drift_px': 0.0, 'n_measured': 2},
        'origin': 1.0,
        'rr': {0: {'n': 2, 'mean_mm': 9.0}},
        'stats': {'n_boxes': 2, 'n_peaks': 1, 'px_per_mm': np.float64(8.0)},
    }


@pytest.fixture
def img(tmp_path):
    p = tmp_path / 'a.png'
    p.write_bytes(b'not really an image, only mtime matters here')
    return str(p)


def test_save_then_load_round_trips(tmp_path, img):
    out = str(tmp_path / 'out')
    cfg = Config()
    rs.save(out, 'D001/a.png', img, cfg, fake_result(), [{'flag': '', 'r_index': 0}])

    got = rs.load(out, 'D001/a.png', image_path=img, cfg=cfg)
    assert got is not None
    assert got['cfg_obj'] == cfg
    assert got['width'] == 60 and got['height'] == 40
    assert got['result']['boxes'] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert got['result']['peaks'][0]['x'] == 1.5
    assert got['result']['grid']['lines'] == [1.0, 41.0]
    assert got['rows'] == [{'flag': '', 'r_index': 0}]
    assert 'raw' not in got['result']            # ภาพต้นฉบับอ่านคืนจากไฟล์เดิมได้


def test_rr_keys_come_back_as_row_numbers(tmp_path, img):
    """json บังคับให้คีย์เป็น string แต่โค้ดที่ใช้ต่อมองหาดัชนีแถวเป็น int"""
    out = str(tmp_path / 'out')
    rs.save(out, 'a.png', img, Config(), fake_result(), [])
    assert rs.load(out, 'a.png')['result']['rr'][0]['mean_mm'] == 9.0


def test_load_returns_none_when_image_changed(tmp_path, img):
    out = str(tmp_path / 'out')
    rs.save(out, 'a.png', img, Config(), fake_result(), [])
    os.utime(img, (0, 0))
    assert rs.load(out, 'a.png', image_path=img) is None


def test_load_returns_none_for_other_config(tmp_path, img):
    out = str(tmp_path / 'out')
    rs.save(out, 'a.png', img, Config(), fake_result(), [])
    assert rs.load(out, 'a.png', cfg=Config().with_(point_pre='gray')) is None
    assert rs.load(out, 'a.png', cfg=Config()) is not None


def test_load_returns_none_for_old_version_or_broken_file(tmp_path, img):
    out = str(tmp_path / 'out')
    rs.save(out, 'a.png', img, Config(), fake_result(), [])
    p = rs.path_for(out, 'a.png')

    payload = json.load(open(p, encoding='utf-8'))
    payload['version'] = rs.VERSION - 1
    json.dump(payload, open(p, 'w', encoding='utf-8'))
    assert rs.load(out, 'a.png') is None

    open(p, 'w', encoding='utf-8').write('{ไม่ใช่ json')
    assert rs.load(out, 'a.png') is None


def test_load_returns_none_when_config_shape_changed(tmp_path, img):
    """ผลที่รันด้วย Config รุ่นอื่นเทียบไม่ได้ ต้องบอกว่าไม่มีผล ไม่ใช่พังหรือเดา"""
    out = str(tmp_path / 'out')
    rs.save(out, 'a.png', img, Config(), fake_result(), [])
    p = rs.path_for(out, 'a.png')
    payload = json.load(open(p, encoding='utf-8'))
    payload['cfg'].pop('point_pre')
    json.dump(payload, open(p, 'w', encoding='utf-8'))
    assert rs.load(out, 'a.png') is None


def test_missing_file_is_not_an_error(tmp_path):
    assert rs.load(str(tmp_path), 'ไม่เคยรัน.png') is None
    assert rs.count(str(tmp_path)) == 0


def test_drop_and_count(tmp_path, img):
    out = str(tmp_path / 'out')
    rs.save(out, 'a.png', img, Config(), fake_result(), [])
    rs.save(out, 'b.png', img, Config(), fake_result(), [])
    assert rs.count(out) == 2
    rs.drop(out, 'a.png')
    assert rs.count(out) == 1 and rs.load(out, 'a.png') is None
    rs.drop(out, 'a.png')                        # ลบซ้ำต้องไม่พัง


def test_names_with_slashes_and_thai_get_distinct_files(tmp_path, img):
    out = str(tmp_path / 'out')
    a = rs.path_for(out, 'D001/681322 ลูน่า 1.jpg')
    b = rs.path_for(out, 'D002/681322 ลูน่า 1.jpg')
    assert a != b and os.sep not in os.path.basename(a)
