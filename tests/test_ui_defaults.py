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
    assert set(options_of('point_pre')) == {'ink', 'gray', 'gray_contrast', 'none'}
    assert set(options_of('point_mode')) == {'refine', 'model_only', 'anchor_only'}
    assert {'train_match', 'mm'} <= set(options_of('crop_mode'))
