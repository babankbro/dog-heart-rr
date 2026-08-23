"""ink mask ต้องตัดกริด (รวมกริดหลักสีแดงเข้ม) และปรับตัวเมื่อเส้นจาง"""
import cv2
import numpy as np
import pytest

from ekg_rpeak.preprocess import (find_r_anchor, ink_mask, mask_quality,
                                  point_preprocess)
from conftest import PPM, make_ekg


def test_anchor_ignores_dark_major_gridlines(cfg, ekg):
    """กริดหลักสีแดงเข้มพาดเต็มความสูง ถ้านับเป็นหมึกจะชนะยอด R เสมอ"""
    img, boxes, truth = ekg
    errs = [abs(find_r_anchor(img, b, cfg)[0] - t) for b, t in zip(boxes, truth)]
    assert max(errs) <= 2, f'anchor คลาดเคลื่อนสูงสุด {max(errs)}px'


def test_ink_mask_excludes_colored_grid(cfg, ekg):
    img, boxes, _ = ekg
    x1, y1, x2, y2 = boxes[1]
    m = ink_mask(img[y1:y2, x1:x2], cfg)
    cols, frac = mask_quality(m)
    assert cols > 0.9, 'เส้นคลื่นควรมีหมึกเกือบทุกคอลัมน์'
    assert frac < 0.25, 'ถ้ากริดหลุดเข้ามา สัดส่วนหมึกจะสูงกว่านี้มาก'


@pytest.mark.parametrize('gray', [35, 150, 195, 225])
def test_adaptive_threshold_finds_faint_traces(cfg, gray):
    """เส้นจางกว่าเพดานเริ่มต้น (160) ต้องยังหาเจอด้วยการปรับ threshold เอง"""
    img, boxes, truth = make_ekg(trace_gray=gray)
    a = find_r_anchor(img, boxes[1], cfg)
    assert a is not None, f'เส้นจาง v={gray} หา anchor ไม่เจอ'
    assert abs(a[0] - truth[1]) <= 2


def test_adaptive_disabled_misses_faint_trace(cfg):
    """ยืนยันว่า adaptive คือสิ่งที่ช่วยจริง ไม่ใช่โชคของ threshold"""
    img, boxes, _ = make_ekg(trace_gray=195)
    assert find_r_anchor(img, boxes[1], cfg.with_(ink_adaptive=False)) is None


def test_blank_roi_returns_none(cfg):
    """ครอปที่ไม่มีเส้นเลย ต้องคืน None ไม่ใช่เดาตำแหน่งมั่ว"""
    img, _, _ = make_ekg(with_trace=False)
    assert find_r_anchor(img, (100, 50, 200, 300), cfg) is None


def test_negative_r_peak(cfg):
    """R ที่เป็นยอดลง (ชี้ลง) ต้องหาเจอเหมือนกัน"""
    img, boxes, truth = make_ekg()
    flipped = cv2.flip(img, 0)
    H = img.shape[0]
    b = boxes[1]
    fb = [b[0], H - b[3], b[2], H - b[1]]
    a = find_r_anchor(flipped, fb, cfg)
    assert a is not None and abs(a[0] - truth[1]) <= 2


def test_point_preprocess_makes_gray(cfg, ekg):
    """ชุดเทรนโมเดลจุดเป็นภาพเทา กริดสีต้องหายไป"""
    img, _, _ = ekg
    out = point_preprocess(img, cfg)
    assert (out[:, :, 0] == out[:, :, 2]).all()
    assert out.min() < 90 and out.max() > 240


def test_point_preprocess_none_passthrough(cfg, ekg):
    img, _, _ = ekg
    assert np.array_equal(point_preprocess(img, cfg.with_(point_pre='none')), img)


def test_anchor_ignores_dotted_gridline(cfg):
    """กริดหลักแบบประที่เข้มพอจะรอดตัวกรองสี ถ้าไม่ตัดทิ้งจะถูกเข้าใจผิดว่าเป็นยอด R

    เป็นอาการที่เจอจริงกับภาพสแกนจริง: จุดประเรียงกันในคอลัมน์เดียว เริ่มที่แถวบนสุด
    จึงชนะยอด R ที่อยู่ต่ำกว่า
    """
    img, boxes, truth = make_ekg()
    b = boxes[1]
    x = int(b[0]) + 3
    for y in range(0, img.shape[0], 3):
        img[y:y + 2, x:x + 2] = (60, 60, 70)       # เทาเข้ม saturation ต่ำ = รอดตัวกรองสี

    a = find_r_anchor(img, b, cfg)
    assert a is not None and abs(a[0] - truth[1]) <= 2, 'anchor ไปเกาะจุดกริดแทนยอด R'

    naive = cfg.with_(ink_keep_trace_only=False, ink_min_run=1)
    assert abs(find_r_anchor(img, b, naive)[0] - truth[1]) > 5, 'เคสนี้ควรทำวิธีเดิมพัง'


def test_keep_trace_drops_isolated_specks(cfg):
    """จุดเล็ก ๆ ที่ไม่ติดกับเส้นคลื่นต้องถูกตัดออกจาก mask"""
    from ekg_rpeak.preprocess import ink_mask, keep_trace
    img, boxes, _ = make_ekg()
    x1, y1, x2, y2 = [int(v) for v in boxes[1]]
    roi = img[y1:y2, x1:x2].copy()
    roi[2:4, 2:4] = (55, 55, 60)
    m = ink_mask(roi, cfg)
    kept = keep_trace(m, cfg)
    assert m[2:4, 2:4].any() and not kept[2:4, 2:4].any()
    assert kept.sum() > 0.8 * m.sum()              # เส้นคลื่นต้องอยู่ครบ


def test_ink_preprocess_removes_grid(cfg, ekg):
    """point_pre='ink' ต้องเหลือแต่เส้นคลื่นดำบนพื้นขาว ไม่มีกริด"""
    img, boxes, _ = ekg
    x1, y1, x2, y2 = [int(v) for v in boxes[2]]
    roi = img[y1:y2, x1:x2]
    out = point_preprocess(roi, cfg.with_(point_pre='ink'))
    vals = np.unique(out)
    assert set(vals.tolist()) <= {0, 255}, 'ต้องเหลือแค่ดำกับขาว'
    ink_frac = float((out[:, :, 0] == 0).mean())
    assert 0.01 < ink_frac < 0.25, f'สัดส่วนหมึก {ink_frac:.3f} ผิดปกติ'
    gray = point_preprocess(roi, cfg.with_(point_pre='gray'))
    mid = ((gray > 20) & (gray < 235)).mean()
    assert mid > 0.05, 'แบบเทาต้องยังมีกริดเป็นสีกลาง ๆ อยู่'
    assert len(np.unique(gray)) > len(vals)      # เทามีหลายระดับกว่าภาพขาวดำล้วน
