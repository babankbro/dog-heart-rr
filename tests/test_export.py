"""CSV ต้องได้ค่าเวลา/bpm ถูกต้อง และติด flag เมื่อจังหวะหลุดหรือซ้ำ"""
import csv

import numpy as np

from ekg_rpeak.export import FIELDS, median_hr, result_to_rows, write_csv
from ekg_rpeak.pipeline import Models, detect_r_peaks
from conftest import BEAT_MM, PPM, FakeCropModel, FakePointModel


def _result(cfg, ekg_path, mode='good'):
    path, boxes, truth = ekg_path
    models = Models(crop=FakeCropModel(boxes), point=FakePointModel(cfg, mode))
    return path, detect_r_peaks(path, models, cfg), truth


def test_bpm_matches_paper_geometry(cfg, ekg_path):
    """RR = 22 mm ที่ 25 mm/s -> 0.88 วินาที -> ~68 bpm"""
    path, r, _ = _result(cfg, ekg_path)
    rows = result_to_rows(path, r, cfg)
    expect = 60.0 / (BEAT_MM / cfg.paper_speed_mm_s)
    bpms = [row['bpm'] for row in rows if row['bpm'] != '']
    assert bpms, 'ไม่มีค่า bpm เลย'
    assert abs(float(np.median(bpms)) - expect) < 2, f'ได้ {np.median(bpms)} คาด {expect:.1f}'
    assert abs(median_hr(r, cfg) - expect) < 2


def test_first_beat_has_no_rr(cfg, ekg_path):
    path, r, _ = _result(cfg, ekg_path)
    rows = result_to_rows(path, r, cfg)
    assert rows[0]['rr_px'] == '' and rows[0]['rr_sec'] == '' and rows[0]['bpm'] == ''
    assert rows[1]['rr_px'] != ''


def test_missed_beat_gets_flagged(cfg, ekg_path):
    """ลบจุดกลางออกหนึ่งจุด RR ช่วงนั้นจะโดดเป็นสองเท่า ต้องติด flag"""
    path, r, _ = _result(cfg, ekg_path)
    r['peaks'].pop(3)
    for i, p in enumerate(r['peaks']):
        p['index'] = i
    rows = result_to_rows(path, r, cfg)
    assert any(row['flag'] == 'missed_beat?' for row in rows)


def test_duplicate_gets_flagged(cfg, ekg_path):
    path, r, _ = _result(cfg, ekg_path)
    p = dict(r['peaks'][2])
    p['x'] += 3.0
    r['peaks'].insert(3, p)
    for i, q in enumerate(r['peaks']):
        q['index'] = i
    rows = result_to_rows(path, r, cfg)
    assert any(row['flag'] == 'duplicate?' for row in rows)


def test_no_scale_leaves_time_columns_empty(cfg, ekg_path):
    """ไม่มีสเกลเลย (ปิดทั้งกริดและการวัดอัตโนมัติ) ต้องเหลือแค่หน่วยพิกเซล"""
    path, boxes, _ = ekg_path
    c = cfg.with_(auto_px_per_mm=False, px_per_mm=None, grid_from_lines=False)
    r = detect_r_peaks(path, Models(crop=FakeCropModel(boxes)), c)
    rows = result_to_rows(path, r, c)
    assert all(row['bpm'] == '' and row['rr_sec'] == '' and row['rr_mm'] == '' for row in rows)
    assert any(row['rr_px'] != '' for row in rows)      # หน่วยพิกเซลยังต้องมี


def test_grid_supplies_scale_without_auto_measure(cfg, ekg_path):
    """ปิดการวัดกริดเล็ก แต่เส้นกริดหลักยังให้สเกลได้เอง"""
    path, boxes, _ = ekg_path
    c = cfg.with_(auto_px_per_mm=False, px_per_mm=None, grid_from_lines=True)
    r = detect_r_peaks(path, Models(crop=FakeCropModel(boxes)), c)
    rows = result_to_rows(path, r, c)
    assert r['stats']['px_per_mm'] and any(row['rr_mm'] != '' for row in rows)


def test_csv_round_trip(cfg, ekg_path, tmp_path):
    path, r, truth = _result(cfg, ekg_path)
    out = str(tmp_path / 'sub' / 'r_peaks.csv')      # โฟลเดอร์ปลายทางต้องถูกสร้างให้
    write_csv(result_to_rows(path, r, cfg), out)
    with open(out, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == FIELDS
    assert len(rows) == len(truth)
    assert rows[0]['src'] == 'model' and rows[0]['cls'] == str(cfg.r_class_id)


def test_gradual_rate_change_is_not_flagged(cfg, ekg_path):
    """หัวใจค่อย ๆ เต้นช้าลงระหว่างบันทึก ต้องไม่ถูกแจ้งว่าจังหวะหาย"""
    path, r, _ = _result(cfg, ekg_path)
    x = 100.0
    step = 60.0
    r['peaks'] = []
    for i in range(20):
        r['peaks'].append({'row': 0, 'index': i, 'x': x, 'y': 50.0,
                           'conf': 0.9, 'src': 'model', 'cls': 5, 'x_mm': x / 8})
        step *= 1.05                       # ช้าลงทีละ 5% ต่อจังหวะ
        x += step
    r['stats']['n_rows'] = 1
    rows = result_to_rows(path, r, cfg)
    assert not any(row['flag'] for row in rows), \
        f"แจ้งผิด: {[row['flag'] for row in rows if row['flag']]}"


def test_single_dropped_beat_is_still_flagged(cfg, ekg_path):
    """จังหวะที่หายจริงกระโดดเป็นสองเท่าเทียบกับเพื่อนบ้าน ต้องยังจับได้"""
    path, r, _ = _result(cfg, ekg_path)
    xs = [100.0 + 80.0 * i for i in range(20)]
    del xs[10]                              # เอาจังหวะกลางออกหนึ่งจุด
    r['peaks'] = [{'row': 0, 'index': i, 'x': x, 'y': 50.0, 'conf': 0.9,
                   'src': 'model', 'cls': 5, 'x_mm': x / 8} for i, x in enumerate(xs)]
    r['stats']['n_rows'] = 1
    rows = result_to_rows(path, r, cfg)
    assert sum(1 for row in rows if row['flag'] == 'missed_beat?') == 1


def test_main_row_is_the_row_with_most_beats():
    """เศษกล่องที่ขอบภาพถูกจัดเป็นแถวของตัวเองได้ ต้องไม่ถูกเข้าใจว่าเป็น lead หลัก"""
    import numpy as np
    from ekg_rpeak.config import Config
    from ekg_rpeak.export import median_hr

    result = {
        'stats': {'px_per_mm': 8.0, 'n_rows': 2},
        'rows': [[np.array([0, 0, 5, 5])], [np.array([0, 0, 5, 5])] * 9],
        'peaks': ([{'row': 0, 'x': 10.0}] +
                  [{'row': 1, 'x': float(100 + 80 * i)} for i in range(9)]),
        'main_row': 1,
    }
    hr = median_hr(result, Config())
    assert hr is not None and 100 < hr < 400        # คิดจากแถวที่ 1 ไม่ใช่แถวที่ 0
    assert median_hr(result, Config(), row=0) is None


def test_main_row_defaults_to_zero_when_absent():
    """ผลรุ่นเก่าที่ยังไม่มี main_row ต้องอ่านได้เหมือนเดิม"""
    from ekg_rpeak.config import Config
    from ekg_rpeak.export import median_hr
    result = {'stats': {'px_per_mm': 8.0, 'n_rows': 1},
              'peaks': [{'row': 0, 'x': 0.0}, {'row': 0, 'x': 80.0}]}
    assert median_hr(result, Config()) is not None
