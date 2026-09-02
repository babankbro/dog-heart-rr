# -*- coding: utf-8 -*-
"""binarization ของภาพที่ป้อนโมเดลครอป และการหายอดที่ไม่ถูกกล่องล็อก"""
import cv2
import numpy as np
import pytest

from conftest import make_ekg
from ekg_rpeak.config import Config
from ekg_rpeak.preprocess import crop_preprocess, find_r_anchor, hysteresis

MODES = ['blackhat', 'blackhat_otsu', 'tophat_gray', 'tophat_red',
         'adaptive', 'ink', 'red']


@pytest.mark.parametrize('mode', MODES)
def test_every_mode_returns_black_trace_on_white_bgr(mode):
    img, _, _ = make_ekg(w=600)
    out = crop_preprocess(img, Config().with_(crop_pre=mode))
    assert out.shape == img.shape and out.dtype == np.uint8
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    assert set(np.unique(g)) <= {0, 255}          # ต้อง binary จริง ไม่ใช่ภาพเทา
    frac = (g < 128).mean()
    assert 0 < frac < 0.6, f'{mode} ให้สัดส่วนเส้น {frac:.2f}'


def test_unknown_mode_is_rejected_loudly():
    img, _, _ = make_ekg(w=200)
    with pytest.raises(ValueError, match='crop_pre'):
        crop_preprocess(img, Config().with_(crop_pre='ไม่มีวิธีนี้'))


def test_blackhat_mode_still_matches_the_training_recipe():
    """โหมด blackhat ต้องเท่ากับสูตรที่ใช้ตอนเทรนเป๊ะ ๆ ห้ามแต่งเพิ่ม"""
    from ekg_rpeak.preprocess import blackhat_preprocess
    img, _, _ = make_ekg(w=400)
    cfg = Config().with_(crop_pre='blackhat')
    assert np.array_equal(crop_preprocess(img, cfg), blackhat_preprocess(img, cfg))


def test_default_mode_is_a_deliberate_departure_from_training():
    """ค่าเริ่มต้นไม่ใช่โดเมนของชุดเทรน — ตั้งใจ เพราะเก็บจังหวะที่ blackhat ทำหลุดได้

    เทสต์นี้มีไว้กันคนเผลอ "แก้กลับ" โดยคิดว่าเป็นบั๊ก ถ้าจะเปลี่ยนต้องวัดผลใหม่ทั้งชุด
    """
    assert Config().crop_pre == 'tophat_red'
    img, _, _ = make_ekg(w=400)
    from ekg_rpeak.preprocess import blackhat_preprocess
    assert not np.array_equal(crop_preprocess(img, Config()), blackhat_preprocess(img, Config()))


def test_anchor_finds_peak_above_a_box_that_cut_the_tip():
    """กล่องจากโมเดลครอปมักตัดยอดทิ้ง anchor ต้องยังหายอดจริงเจอ

    จำลองอาการที่วัดได้จากภาพจริง: ขอบบนของกล่องอยู่ต่ำกว่ายอด R หลายสิบพิกเซล
    เพราะโมเดลครอปเห็นภาพที่ผ่าน binarization ซึ่งกินปลายยอดที่หมึกจางไปแล้ว
    """
    img, boxes, _ = make_ekg(w=800)
    x1, y1, x2, y2 = (int(v) for v in boxes[1])
    no_expand = Config().with_(anchor_expand_y=0.0)

    full = find_r_anchor(img, (x1, y1, x2, y2), no_expand)
    assert full is not None

    cut = (x1, y1 + 40, x2, y2)                      # กล่องที่ตัดยอดทิ้ง
    tight = find_r_anchor(img, cut, no_expand)
    loose = find_r_anchor(img, cut, Config())

    assert tight is not None and tight[1] >= y1 + 40 - 1   # ติดขอบกล่อง ไม่ใช่ยอดจริง
    assert loose is not None and abs(loose[1] - full[1]) < 10   # ขยายแล้วได้ยอดเดิมคืน


def test_vertical_expansion_is_on_by_default():
    assert Config().anchor_expand_y > 0


# ---------------------------------------------------------------- แชนเนลแดง / hysteresis

def trace_mask(img, cfg):
    return cv2.cvtColor(crop_preprocess(img, cfg), cv2.COLOR_BGR2GRAY) < 128


def test_red_channel_removes_the_grid_that_blackhat_keeps():
    """กริดพิมพ์ด้วยหมึกแดง ในแชนเนลแดงจึงสว่างเกือบเท่ากระดาษ"""
    img, _, _ = make_ekg(w=800)
    bh = trace_mask(img, Config().with_(crop_pre='blackhat')).mean()
    red = trace_mask(img, Config().with_(crop_pre='red')).mean()
    assert red < bh / 2, f'red={red:.3f} ไม่ได้สะอาดกว่า blackhat={bh:.3f}'


def test_dilate_thickens_the_trace():
    img, _, _ = make_ekg(w=800)
    thin = trace_mask(img, Config().with_(crop_pre='red')).sum()
    thick = trace_mask(img, Config().with_(crop_pre='red', crop_pre_dilate=3)).sum()
    assert thick > thin


def test_close_heals_a_broken_trace():
    """เส้นที่สแกนมาจางเป็นช่วง ๆ ทำให้ binarize แล้วเส้นขาด"""
    img, _, _ = make_ekg(w=800, trace_gray=35)
    img[:, 300:306] = 255                      # เจาะช่องว่างกลางเส้น
    broken = trace_mask(img, Config().with_(crop_pre='red'))
    healed = trace_mask(img, Config().with_(crop_pre='red', crop_pre_close=9))

    def gap_cols(m):
        has = m.any(axis=0)
        idx = np.where(has)[0]
        return int((~has[idx[0]:idx[-1] + 1]).sum()) if idx.size > 1 else -1

    assert gap_cols(healed) < gap_cols(broken)


def test_hysteresis_keeps_only_what_touches_the_core():
    strong = np.zeros((10, 10), bool)
    strong[5, 5] = True
    weak = np.zeros((10, 10), bool)
    weak[4:7, 5] = True          # ติดกับแกน -> ต้องเก็บ
    weak[0, 0] = True            # จุดลอย ๆ ไกลออกไป -> ต้องทิ้ง
    out = hysteresis(strong, weak)
    assert out[4:7, 5].all() and not out[0, 0]


def test_hysteresis_on_empty_core_returns_nothing():
    empty = np.zeros((5, 5), bool)
    weak = np.ones((5, 5), bool)
    assert not hysteresis(empty, weak).any()


def test_hysteresis_recovers_more_of_a_faint_trace():
    img, _, _ = make_ekg(w=800, trace_gray=35)
    plain = trace_mask(img, Config().with_(crop_pre='red')).sum()
    hyst = trace_mask(img, Config().with_(crop_pre='red', crop_pre_hyst=0.5)).sum()
    assert hyst >= plain


def test_blackhat_ignores_the_post_processing_knobs():
    """โหมดที่ตรงกับชุดเทรนต้องไม่ถูกแต่งเพิ่ม ไม่งั้นหลุดโดเมนโดยไม่ตั้งใจ"""
    img, _, _ = make_ekg(w=400)
    a = crop_preprocess(img, Config().with_(crop_pre='blackhat'))
    b = crop_preprocess(img, Config().with_(crop_pre='blackhat', crop_pre_dilate=5,
                                            crop_pre_close=7, crop_pre_hyst=0.9))
    assert np.array_equal(a, b)


# ---------------------------------------------------------------- tophat บนแชนเนลแดง

def test_tophat_red_keeps_less_grid_than_tophat_gray():
    """เส้นกริดหลักรอด tophat_gray มาเป็นแท่งทึบ ทำให้ยอด R ที่พาดทับกริดกลืนหาย

    ทำ tophat บนแชนเนลแดงแทน กริดหมึกแดงจึงเกือบไม่ต่างจากกระดาษตั้งแต่แรก
    """
    img, _, _ = make_ekg(w=900, major_grid=True)
    grayish = trace_mask(img, Config().with_(crop_pre='tophat_gray')).mean()
    reddish = trace_mask(img, Config().with_(crop_pre='tophat_red')).mean()
    assert reddish < grayish


def test_tophat_red_still_keeps_the_trace():
    """ลบกริดแล้วเส้นคลื่นต้องยังอยู่ ไม่ใช่ลบเกลี้ยง"""
    img, _, _ = make_ekg(w=900)
    m = trace_mask(img, Config().with_(crop_pre='tophat_red'))
    assert 0.001 < m.mean() < 0.3
    blank = np.full_like(img, 255)
    assert trace_mask(blank, Config().with_(crop_pre='tophat_red')).mean() < 0.01


def test_tophat_modes_react_to_their_own_knobs():
    img, _, _ = make_ekg(w=600)
    for mode in ('tophat_gray', 'tophat_red'):
        base = trace_mask(img, Config().with_(crop_pre=mode)).sum()
        loose = trace_mask(img, Config().with_(crop_pre=mode, crop_pre_thr=5)).sum()
        assert loose >= base, mode


# ---------------------------------------------------------------- preprocessing ของโมเดลจุด

def point_mask(img, cfg):
    from ekg_rpeak.preprocess import point_preprocess
    return cv2.cvtColor(point_preprocess(img, cfg), cv2.COLOR_BGR2GRAY) < 128


def test_every_point_mode_returns_a_usable_image():
    from ekg_rpeak.preprocess import POINT_PRE_MODES, point_preprocess
    img, _, _ = make_ekg(w=500)
    for mode in POINT_PRE_MODES:
        out = point_preprocess(img, Config().with_(point_pre=mode))
        assert out.shape == img.shape and out.dtype == np.uint8, mode


def test_red_ink_removes_more_grid_than_ink():
    """กริดพิมพ์หมึกแดง แชนเนลแดงจึงตัดได้ตรงกว่าการดูที่ saturation"""
    img, _, _ = make_ekg(w=700, major_grid=True)
    assert point_mask(img, Config().with_(point_pre='red_ink')).mean() <=            point_mask(img, Config().with_(point_pre='ink')).mean() + 1e-9


def test_red_ink_keeps_the_trace():
    img, _, _ = make_ekg(w=700)
    assert 0.001 < point_mask(img, Config().with_(point_pre='red_ink')).mean() < 0.3


def test_stretch_leaves_flat_images_alone():
    """ครอปที่แทบไม่มีคอนทราสต์ ห้ามถูกดึงจนกลายเป็นสัญญาณรบกวน"""
    from ekg_rpeak.preprocess import stretch
    flat = np.full((40, 40), 200, np.uint8)
    assert np.array_equal(stretch(flat), flat)
    ramp = np.tile(np.linspace(20, 230, 40).astype(np.uint8), (40, 1))
    out = stretch(ramp)
    assert out.min() < ramp.min() + 5 and out.max() > ramp.max() - 5


def test_default_point_mode_is_the_measured_winner():
    """กันคนเผลอแก้กลับ — เปลี่ยนแล้วต้องวัดกับทั้งชุดใหม่"""
    assert Config().point_pre == 'red_ink'
