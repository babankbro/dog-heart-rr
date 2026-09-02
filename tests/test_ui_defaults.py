"""หน้าเว็บต้องส่งค่าตั้งตรงกับค่าเริ่มต้นของเซิร์ฟเวอร์

บั๊กที่เคยเกิด: loadDefaults เขียนค่าลงช่องเฉพาะเมื่อช่องว่าง แต่ <select>
ไม่มีวันว่าง มันเลือก option แรกเสมอ หน้าเว็บจึงส่ง point_pre=gray กับ
crop_mode=mm ทับค่าเริ่มต้น ทำให้โมเดล ink ทำนายไม่ได้เลย (model=0)
"""
import os
import re

import pytest

from ekg_rpeak.config import Config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, 'webapp', 'static', 'index.html'), encoding='utf-8').read()
JS = open(os.path.join(ROOT, 'webapp', 'static', 'app.js'), encoding='utf-8').read()


def options_of(select_id):
    m = re.search(rf'<select id="{select_id}">(.*?)</select>', HTML, re.S)
    assert m, f'ไม่พบ select id={select_id}'
    return re.findall(r'<option value="([^"]+)"', m.group(1))


@pytest.mark.parametrize('field', ['crop_mode', 'point_pre', 'point_mode'])
def test_first_option_matches_server_default(field):
    """option แรกต้องเป็นค่าเริ่มต้นจริง เผื่อกรณีที่ดึง config ไม่สำเร็จ"""
    opts = options_of(field)
    assert opts[0] == getattr(Config(), field), \
        f'option แรกของ {field} คือ {opts[0]} แต่ค่าเริ่มต้นคือ {getattr(Config(), field)}'


@pytest.mark.parametrize('field', ['crop_mode', 'point_pre', 'point_mode'])
def test_server_default_exists_as_option(field):
    assert getattr(Config(), field) in options_of(field)


def test_load_defaults_assigns_select_values():
    """loadDefaults ต้องเซ็ตค่าให้ select ไม่ใช่เช็คแค่ว่าช่องว่างหรือไม่"""
    m = re.search(r'async function loadDefaults\(\)\s*\{(.*?)\n\}', JS, re.S)
    assert m, 'ไม่พบ loadDefaults'
    body = m.group(1)
    assert "tagName === 'SELECT'" in body, 'ต้องแยกกรณี select ออกมาจัดการต่างหาก'
    assert re.search(r"el\.value\s*=\s*String\(v\)", body), 'ต้องกำหนดค่าให้ select โดยตรง'


def test_options_list_covers_config_choices():
    """ตัวเลือกในหน้าเว็บต้องไม่หลุดจากที่ไปป์ไลน์รองรับ"""
    from ekg_rpeak.preprocess import POINT_PRE_MODES
    assert set(options_of('point_pre')) == set(POINT_PRE_MODES)
    assert set(options_of('point_mode')) == {'refine', 'model_only', 'anchor_only'}
    assert {'train_match', 'mm'} <= set(options_of('crop_mode'))


def test_crop_pre_select_matches_server_default():
    """เพิ่ม select ใหม่แล้วต้องไม่ทำบั๊กเดิมซ้ำ: option แรกต้องเป็นค่าเริ่มต้นจริง"""
    opts = options_of('crop_pre')
    assert opts[0] == Config().crop_pre
    from ekg_rpeak.preprocess import CROP_PRE_MODES
    assert set(opts) == set(CROP_PRE_MODES)


def test_binarization_knobs_are_wired_to_the_preview():
    """ปรับค่าแล้วภาพต้องอัปเดตเอง ไม่ใช่ต้องกดรันใหม่"""
    assert 'PRE_OPTS' in JS
    assert re.search(r"for \(const k of PRE_OPTS\) \$\(k\)\.onchange = refreshPrebin", JS)
    for k in ('crop_pre', 'blackhat_thr', 'crop_pre_hyst', 'crop_pre_close', 'crop_pre_dilate'):
        assert f'id="{k}"' in HTML, k


def test_tophat_knobs_are_in_both_pages():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dbg = open(os.path.join(root, 'webapp', 'static', 'debug.html'), encoding='utf-8').read()
    for k in ('crop_pre_thr', 'crop_pre_ksize'):
        assert f'id="{k}"' in HTML, k
        assert dbg.count(f'data-k="{k}"') == 2, k
        assert k in JS


def test_pages_without_a_sidebar_clear_the_grid_layout():
    """main เป็น grid 300px + 1fr สำหรับหน้าหลัก หน้าที่ไม่มีแถบข้างต้องล้างทิ้ง

    ถ้าไม่ล้าง เนื้อหาของหน้านั้นจะไปตกในคอลัมน์ 300px ตารางจึงแคบจนอ่านไม่ได้
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    css = open(os.path.join(root, 'webapp', 'static', 'style.css'), encoding='utf-8').read()
    assert re.search(r'main\.dbg\s*\{[^}]*display:\s*block', css)
    for page in ('rr.html', 'debug.html'):
        html = open(os.path.join(root, 'webapp', 'static', page), encoding='utf-8').read()
        assert re.search(r'<main[^>]*class="[^"]*\bdbg\b', html), page


def debug_select_options(field):
    """ตัวเลือกของ select บนหน้า debug — มีสองชุด (ฝั่งเทรนกับฝั่งเทส) ต้องเหมือนกัน"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html = open(os.path.join(root, 'webapp', 'static', 'debug.html'), encoding='utf-8').read()
    blocks = re.findall(rf'<select data-k="{field}">(.*?)</select>', html, re.S)
    assert len(blocks) == 2, f'{field} ควรมีสองชุด พบ {len(blocks)}'
    opts = [re.findall(r'<option value="([^"]+)"', b) for b in blocks]
    assert opts[0] == opts[1], f'{field} สองฝั่งมีตัวเลือกไม่ตรงกัน'
    return opts[0]


@pytest.mark.parametrize('field', ['crop_pre', 'point_pre'])
def test_debug_page_offers_every_mode_the_pipeline_supports(field):
    """หน้า debug ต้องมีทุกโหมด ไม่งั้นเลือกวิธีที่โมเดลใช้อยู่จริงไม่ได้"""
    from ekg_rpeak import preprocess as pp
    modes = pp.CROP_PRE_MODES if field == 'crop_pre' else pp.POINT_PRE_MODES
    assert set(debug_select_options(field)) == set(modes)


@pytest.mark.parametrize('field', ['crop_pre', 'point_pre'])
def test_debug_page_defaults_to_what_the_model_actually_uses(field):
    """option แรกต้องเป็นค่าเริ่มต้นจริง

    บั๊กที่เคยเกิด: หน้า debug ไม่มี red_ink ในรายการ loadDefaults จึงเซ็ตค่าไม่ได้
    หน้านั้นเลยรันด้วย ink เงียบ ๆ ทั้งที่โมเดลใช้ red_ink ผลที่เอามาเทียบจึงคนละวิธี
    """
    assert debug_select_options(field)[0] == getattr(Config(), field)


def test_table_tab_offers_a_whole_patient_download():
    """ปุ่มรวมต้องผูกกับสัตว์ที่เลือก ไม่ใช่ภาพที่กำลังดู"""
    assert 'id="csvAllLink"' in HTML
    assert 'setPatientCsv' in JS
    assert re.search(r"/api/patients/\$\{encodeURIComponent\(pid\)\}/csv", JS)
