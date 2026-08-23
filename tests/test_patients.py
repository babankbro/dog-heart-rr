"""ทะเบียนสัตว์: สร้าง แก้ไข ลบ จัดการภาพ และการจัดกลุ่มภาพเดิม"""
import os

import cv2
import pytest

from ekg_rpeak import patients as pt
from conftest import make_ekg


@pytest.fixture
def data(tmp_path):
    d = tmp_path / 'data'
    d.mkdir()
    return str(d)


def png_bytes(w=400):
    img, _, _ = make_ekg(w=w)
    return cv2.imencode('.png', img)[1].tobytes()


def test_create_and_list(data):
    p = pt.create_patient(data, 'D001', 'Buddy', 'บีเกิล 3 ปี')
    assert p['id'] == 'D001' and p['name'] == 'Buddy' and p['n_images'] == 0
    assert p['created']
    lst = pt.list_patients(data)
    assert [x['id'] for x in lst] == ['D001']
    assert os.path.isdir(pt.patient_dir(data, 'D001'))


def test_create_rejects_bad_id(data):
    for bad in ('', '   ', '../evil', 'a' * 40, 'มี ช่องว่าง'):
        with pytest.raises(ValueError):
            pt.create_patient(data, bad)


def test_create_duplicate(data):
    pt.create_patient(data, 'C003')
    with pytest.raises(FileExistsError):
        pt.create_patient(data, 'C003')


def test_update_patient(data):
    pt.create_patient(data, 'D002', 'ชื่อเดิม')
    out = pt.update_patient(data, 'D002', name='Mali', note='โน้ต')
    assert out['name'] == 'Mali' and out['note'] == 'โน้ต'
    assert pt.get_patient(data, 'D002')['name'] == 'Mali'


def test_add_and_list_images(data):
    pt.create_patient(data, 'A1')
    n1 = pt.add_image(data, 'A1', 'first.png', png_bytes())
    n2 = pt.add_image(data, 'A1', 'second.png', png_bytes())
    assert n1 == 'A1/first.png' and n2 == 'A1/second.png'
    assert pt.list_patient_images(data, 'A1') == [n1, n2]
    assert pt.get_patient(data, 'A1')['n_images'] == 2


def test_add_image_does_not_overwrite(data):
    pt.create_patient(data, 'A1')
    a = pt.add_image(data, 'A1', 'x.png', png_bytes())
    b = pt.add_image(data, 'A1', 'x.png', png_bytes())
    assert a != b and b.endswith('x_1.png')       # ไฟล์เดิมต้องไม่ถูกทับ


def test_add_image_rejects_bad_input(data):
    pt.create_patient(data, 'A1')
    with pytest.raises(ValueError):
        pt.add_image(data, 'A1', 'note.txt', b'hello')
    with pytest.raises(ValueError):
        pt.add_image(data, 'A1', 'broken.png', b'not an image')
    assert pt.list_patient_images(data, 'A1') == []   # ไฟล์เสียต้องไม่ค้างอยู่


def test_delete_image(data):
    pt.create_patient(data, 'A1')
    name = pt.add_image(data, 'A1', 'x.png', png_bytes())
    pt.delete_image(data, 'A1', 'x.png')
    assert pt.list_patient_images(data, 'A1') == []
    with pytest.raises(FileNotFoundError):
        pt.delete_image(data, 'A1', 'x.png')


def test_delete_image_blocks_path_traversal(data, tmp_path):
    pt.create_patient(data, 'A1')
    outside = tmp_path / 'secret.png'
    outside.write_bytes(png_bytes())
    with pytest.raises(FileNotFoundError):
        pt.delete_image(data, 'A1', '../secret.png')
    assert outside.exists()


def test_delete_patient_keeps_images_by_default(data):
    pt.create_patient(data, 'A1')
    pt.add_image(data, 'A1', 'x.png', png_bytes())
    pt.delete_patient(data, 'A1')
    assert os.path.isdir(pt.patient_dir(data, 'A1'))       # ภาพยังอยู่
    assert [p['id'] for p in pt.list_patients(data)] == ['A1']   # โฟลเดอร์ยังโผล่ในรายการ

    pt.delete_patient(data, 'A1', with_images=True)
    assert not os.path.isdir(pt.patient_dir(data, 'A1'))
    assert pt.list_patients(data) == []


def test_folder_without_index_still_listed(data):
    """โฟลเดอร์ที่ก๊อปมาวางเองโดยไม่มีในดัชนี ต้องยังเห็นในรายการ"""
    os.makedirs(os.path.join(data, 'Z9'))
    cv2.imwrite(os.path.join(data, 'Z9', 'a.png'), make_ekg(w=400)[0])
    lst = pt.list_patients(data)
    assert [p['id'] for p in lst] == ['Z9'] and lst[0]['n_images'] == 1


@pytest.mark.parametrize('fname,pid,name', [
    ('D001 Buddy 1.jpg', 'D001', 'Buddy'),        # มีเว้นวรรคก่อนเลขลำดับ
    ('D002 Mali2.jpg', 'D002', 'Mali'),           # ไม่มีเว้นวรรค
    ('C003 ส้มโอ 1.jpg', 'C003', 'ส้มโอ'),          # ชื่อภาษาไทย
])
def test_parse_filename(fname, pid, name):
    assert pt.parse_filename(fname) == (pid, name)


def test_parse_filename_rejects_unparsable():
    assert pt.parse_filename('scan.jpg') is None


def test_migrate_groups_flat_images(data):
    for fn in ('D001 Buddy 1.jpg', 'D001 Buddy2.jpg', 'D002 Mali1.jpg'):
        cv2.imwrite(os.path.join(data, fn), make_ekg(w=400)[0])
    cv2.imwrite(os.path.join(data, 'scan.jpg'), make_ekg(w=400)[0])   # แกะรหัสไม่ได้

    res = pt.migrate_flat_images(data)
    assert len(res['moved']) == 3 and res['skipped'] == ['scan.jpg']
    assert os.path.exists(os.path.join(data, 'scan.jpg'))             # ต้องไม่ถูกย้าย

    lst = {p['id']: p for p in pt.list_patients(data)}
    assert lst['D001']['n_images'] == 2 and lst['D001']['name'] == 'Buddy'
    assert lst['D002']['n_images'] == 1 and lst['D002']['name'] == 'Mali'


def test_migrate_is_idempotent(data):
    cv2.imwrite(os.path.join(data, 'D001 Buddy 1.jpg'), make_ekg(w=400)[0])
    pt.migrate_flat_images(data)
    again = pt.migrate_flat_images(data)
    assert again['moved'] == []                    # รันซ้ำต้องไม่ย้ายอะไรอีก
    assert pt.get_patient(data, 'D001')['n_images'] == 1


def test_index_survives_corrupt_file(data):
    pt.create_patient(data, 'A1', 'ชื่อ')
    with open(pt.index_path(data), 'w', encoding='utf-8') as f:
        f.write('{ไม่ใช่ json')
    assert pt.load_index(data) == {}               # ไม่พังทั้งระบบ
    assert [p['id'] for p in pt.list_patients(data)] == ['A1']   # ยังเห็นจากโฟลเดอร์
