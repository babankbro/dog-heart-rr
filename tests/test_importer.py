# -*- coding: utf-8 -*-
"""ดึงภาพจากโครงโฟลเดอร์ภายนอกเข้าทะเบียนสัตว์

ต้นทางเป็นข้อมูลที่คนจัดเอง เทสต์จึงจำลองความเพี้ยนที่เจอจริงในชุด MSU
"""
import os

import cv2
import pytest

from conftest import make_ekg
from ekg_rpeak import patients as pt
from ekg_rpeak.importer import clean_group, image_ext, import_tree, split_folder_name


def put(path, name, ext='.jpg'):
    """วางไฟล์ภาพจริงหนึ่งใบ ชื่อไฟล์อาจไม่มีนามสกุลก็ได้"""
    os.makedirs(path, exist_ok=True)
    img, _, _ = make_ekg(w=200)
    ok, buf = cv2.imencode(ext, img)
    assert ok
    with open(os.path.join(path, name), 'wb') as f:
        f.write(buf.tobytes())


@pytest.fixture
def src(tmp_path):
    r = tmp_path / 'src'
    put(str(r / '1. Normal' / 'D001 Buddy' / 'crop'), 'a.jpg')
    put(str(r / '1. Normal' / 'D001 Buddy' / 'crop'), 'b.jpg')
    put(str(r / '1. Normal' / 'D001 Buddy' / 'ori'), 'orig.jpg')
    put(str(r / '1. Normal' / 'D002 Coco' / 'Crop'), 'x.jpg')        # C ใหญ่
    put(str(r / '2. B1 LA-Ao' / 'D003' / 'crop'), 'ไฟล์ - 2026Z')     # ไม่มีนามสกุล
    return str(r)


def test_clean_group_strips_leading_number():
    assert clean_group('1. Normal') == 'Normal'
    assert clean_group('4.LA-Ao _ 1.6') == 'LA-Ao _ 1.6'
    assert clean_group('Normal') == 'Normal'


def test_split_folder_name():
    assert split_folder_name('D001 Buddy') == ('D001', 'Buddy')
    assert split_folder_name('681286') == ('681286', '')
    assert split_folder_name('c64515 หลืบศรี-ลูซี่') == ('c64515', 'หลืบศรี-ลูซี่')
    assert split_folder_name('../evil ชื่อ') is None


def test_image_ext_reads_bytes_not_filename(tmp_path):
    put(str(tmp_path), 'ไม่มีนามสกุล')
    assert image_ext(str(tmp_path / 'ไม่มีนามสกุล')) == '.jpg'
    (tmp_path / 'ปลอม.jpg').write_bytes(b'not an image')
    assert image_ext(str(tmp_path / 'ปลอม.jpg')) is None


def test_import_tree_groups_by_folder_and_takes_crop_only(src, tmp_path):
    data = str(tmp_path / 'data')
    os.makedirs(data)
    r = import_tree(src, data)

    assert r['n_patients'] == 3 and r['n_images'] == 4      # ori ไม่ถูกดึงมา
    by_id = {p['id']: p for p in pt.list_patients(data)}
    assert by_id['D001']['group'] == 'Normal' and by_id['D001']['n_images'] == 2
    assert by_id['D002']['group'] == 'Normal'               # โฟลเดอร์ "Crop" ก็ต้องเจอ
    assert by_id['D003']['group'] == 'B1 LA-Ao'
    assert pt.list_groups(data) == ['B1 LA-Ao', 'Normal']


def test_import_tree_renames_files_readably(src, tmp_path):
    """ชื่อไฟล์ต้นทางอ่านไม่รู้เรื่องและบางไฟล์ไม่มีนามสกุล ต้องตั้งใหม่ให้ใช้งานได้"""
    data = str(tmp_path / 'data')
    os.makedirs(data)
    import_tree(src, data)
    assert sorted(os.listdir(os.path.join(data, 'D001'))) == ['D001 Buddy 1.jpg', 'D001 Buddy 2.jpg']
    assert os.listdir(os.path.join(data, 'D003')) == ['D003 1.jpg']
    assert pt.list_patient_images(data, 'D001') == ['D001/D001 Buddy 1.jpg', 'D001/D001 Buddy 2.jpg']


def test_import_tree_limit_caps_images_per_patient(src, tmp_path):
    data = str(tmp_path / 'data')
    os.makedirs(data)
    r = import_tree(src, data, limit=1)
    assert r['n_images'] == 3 and len(os.listdir(os.path.join(data, 'D001'))) == 1


def test_import_tree_replace_removes_old_images(src, tmp_path):
    data = str(tmp_path / 'data')
    put(os.path.join(data, 'D001'), 'ของเก่า.jpg')
    import_tree(src, data, replace=True)
    assert 'ของเก่า.jpg' not in os.listdir(os.path.join(data, 'D001'))

    put(os.path.join(data, 'D001'), 'ของเก่า.jpg')
    import_tree(src, data)                                   # ไม่ replace = เก็บของเดิมไว้
    assert 'ของเก่า.jpg' in os.listdir(os.path.join(data, 'D001'))


def test_import_tree_reports_what_it_skipped(tmp_path):
    r = tmp_path / 'src'
    (r / '1. Normal' / 'D001 Buddy' / 'ori').mkdir(parents=True)   # ไม่มีโฟลเดอร์ crop
    (r / '1. Normal' / 'D002 Coco' / 'crop').mkdir(parents=True)   # crop ว่าง
    data = str(tmp_path / 'data')
    os.makedirs(data)
    out = import_tree(str(r), data)
    assert out['n_patients'] == 0
    reasons = ' | '.join(s['reason'] for s in out['skipped'])
    assert 'crop' in reasons and 'ไม่มีไฟล์ภาพ' in reasons


def test_import_tree_rejects_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        import_tree(str(tmp_path / 'ไม่มีจริง'), str(tmp_path))


def test_import_tree_replaces_id_that_differs_only_by_case(src, tmp_path):
    """C56153 กับ c56153 คือตัวเดียวกันบน Windows ทะเบียนต้องเหลือระเบียนเดียว"""
    data = str(tmp_path / 'data')
    os.makedirs(data)
    pt.create_patient(data, 'D001'.upper(), 'ชื่อเก่า', group='ประเภทเก่า')
    import_tree(src, data, replace=True)

    ids = [p['id'] for p in pt.list_patients(data)]
    assert ids.count('D001') == 1 and len(ids) == len(set(i.lower() for i in ids))
    assert pt.get_patient(data, 'D001')['group'] == 'Normal'
