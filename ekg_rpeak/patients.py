"""ทะเบียนสัตว์และภาพของแต่ละตัว

ภาพถูกเก็บแยกโฟลเดอร์ตามรหัสสัตว์ (data/<id>/...) และมีไฟล์ดัชนี patients.json
เก็บชื่อกับหมายเหตุ ทำให้ภาพหลายใบของตัวเดียวกันถูกรวมเป็นหน้าเดียวได้

รหัสสัตว์มาจากชื่อไฟล์เดิมที่ขึ้นต้นด้วยรหัส เช่น "D001 Buddy 1.jpg" -> D001
"""
import json
import os
import re
import shutil
from datetime import date
from typing import Dict, List, Optional

from .imageio import imread_u, list_images

INDEX_NAME = 'patients.json'
# ฟิลด์ที่ทุกระเบียนต้องมี พร้อมค่าเริ่มต้นสำหรับดัชนีเก่าที่ยังไม่มีฟิลด์นี้
FIELD_DEFAULTS = {'name': '', 'note': '', 'group': '', 'created': ''}
ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,32}$')
# ชื่อไฟล์รูปแบบ "<รหัส> <ชื่อ><ลำดับ>.jpg" เช่น "D001 Buddy 1.jpg", "C003 Snow 2.jpg"
NAME_RE = re.compile(r'^\s*([A-Za-z0-9_-]+)\s+(.*?)\s*(\d+)?\s*$')


def index_path(data_dir: str) -> str:
    return os.path.join(data_dir, INDEX_NAME)


def load_index(data_dir: str) -> Dict[str, dict]:
    p = index_path(data_dir)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding='utf-8') as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    return {p['id']: {**FIELD_DEFAULTS, **p}
            for p in raw.get('patients', []) if p.get('id')}


def save_index(data_dir: str, patients: Dict[str, dict]) -> None:
    os.makedirs(data_dir, exist_ok=True)
    tmp = index_path(data_dir) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump({'patients': sorted(patients.values(), key=lambda p: p['id'])},
                  f, ensure_ascii=False, indent=1)
    os.replace(tmp, index_path(data_dir))     # เขียนแบบ atomic กันไฟล์ดัชนีพัง


def valid_id(pid: str) -> bool:
    return bool(pid) and bool(ID_RE.match(pid))


def patient_dir(data_dir: str, pid: str) -> str:
    return os.path.join(data_dir, pid)


def parse_filename(fname: str):
    """แยกรหัสกับชื่อออกจากชื่อไฟล์เดิม คืน (id, name) หรือ None"""
    stem = os.path.splitext(os.path.basename(fname))[0]
    m = NAME_RE.match(stem)
    if not m:
        return None
    pid, name = m.group(1), (m.group(2) or '').strip()
    return (pid, name) if valid_id(pid) else None


def list_patient_images(data_dir: str, pid: str) -> List[str]:
    """ชื่อไฟล์ภาพของสัตว์ตัวนี้ (relative กับ data_dir เพื่อใช้กับ API เดิมได้)"""
    d = patient_dir(data_dir, pid)
    if not os.path.isdir(d):
        return []
    return [os.path.relpath(p, data_dir).replace(os.sep, '/')
            for p in list_images(d)]


def list_patients(data_dir: str) -> List[dict]:
    """รวมข้อมูลจากดัชนีกับโฟลเดอร์จริง โฟลเดอร์ที่ไม่มีในดัชนีก็ยังแสดง"""
    idx = load_index(data_dir)
    seen = dict(idx)
    if os.path.isdir(data_dir):
        for name in sorted(os.listdir(data_dir)):
            d = os.path.join(data_dir, name)
            if os.path.isdir(d) and valid_id(name) and name not in seen:
                seen[name] = {'id': name, **FIELD_DEFAULTS}
    out = []
    for pid, rec in sorted(seen.items()):
        imgs = list_patient_images(data_dir, pid)
        out.append({**FIELD_DEFAULTS, **rec, 'id': pid,
                    'n_images': len(imgs), 'images': imgs})
    return out


def list_groups(data_dir: str) -> List[str]:
    """ประเภทที่มีสัตว์อยู่จริง เรียงตามตัวอักษร ใช้ทำตัวกรองในหน้าเว็บ"""
    return sorted({p['group'] for p in list_patients(data_dir) if p.get('group')})


def get_patient(data_dir: str, pid: str) -> Optional[dict]:
    for p in list_patients(data_dir):
        if p['id'] == pid:
            return p
    return None


def create_patient(data_dir: str, pid: str, name: str = '', note: str = '',
                   group: str = '') -> dict:
    if not valid_id(pid):
        raise ValueError(f'รหัสไม่ถูกต้อง: {pid!r} (ใช้ได้เฉพาะ A-Z a-z 0-9 _ - ไม่เกิน 32 ตัว)')
    idx = load_index(data_dir)
    if pid in idx or os.path.isdir(patient_dir(data_dir, pid)):
        raise FileExistsError(f'มีรหัส {pid} อยู่แล้ว')
    os.makedirs(patient_dir(data_dir, pid), exist_ok=True)
    idx[pid] = {'id': pid, 'name': name.strip(), 'note': note.strip(),
                'group': group.strip(), 'created': date.today().isoformat()}
    save_index(data_dir, idx)
    return {**idx[pid], 'n_images': 0, 'images': []}


def update_patient(data_dir: str, pid: str, name=None, note=None, group=None) -> dict:
    idx = load_index(data_dir)
    rec = idx.get(pid) or {'id': pid, **FIELD_DEFAULTS}
    for key, val in (('name', name), ('note', note), ('group', group)):
        if val is not None:
            rec[key] = val.strip()
    idx[pid] = rec
    save_index(data_dir, idx)
    return get_patient(data_dir, pid)


def delete_patient(data_dir: str, pid: str, with_images: bool = False) -> None:
    """ลบออกจากทะเบียน ถ้า with_images จะลบโฟลเดอร์ภาพด้วย"""
    idx = load_index(data_dir)
    idx.pop(pid, None)
    save_index(data_dir, idx)
    d = patient_dir(data_dir, pid)
    if with_images and os.path.isdir(d):
        shutil.rmtree(d)


def add_image(data_dir: str, pid: str, filename: str, blob: bytes) -> str:
    """บันทึกภาพเข้าโฟลเดอร์ของสัตว์ คืนชื่อไฟล์แบบ relative"""
    if not valid_id(pid):
        raise ValueError(f'รหัสไม่ถูกต้อง: {pid!r}')
    d = patient_dir(data_dir, pid)
    os.makedirs(d, exist_ok=True)
    base = os.path.basename(filename or 'image.png')
    if not base.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise ValueError('รองรับเฉพาะ .jpg .jpeg .png')
    stem, ext = os.path.splitext(base)
    dest, k = os.path.join(d, base), 1
    while os.path.exists(dest):                   # กันชื่อชนกัน ไม่ทับของเดิม
        dest = os.path.join(d, f'{stem}_{k}{ext}')
        k += 1
    with open(dest, 'wb') as f:
        f.write(blob)
    if imread_u(dest) is None:
        os.remove(dest)
        raise ValueError('ไฟล์นี้เปิดเป็นภาพไม่ได้')
    return os.path.relpath(dest, data_dir).replace(os.sep, '/')


def delete_image(data_dir: str, pid: str, name: str) -> None:
    """ลบภาพหนึ่งใบ กันไม่ให้ path หลุดออกนอกโฟลเดอร์ของสัตว์ตัวนั้น"""
    d = os.path.abspath(patient_dir(data_dir, pid))
    p = os.path.abspath(os.path.join(d, os.path.basename(name)))
    if not p.startswith(d + os.sep) or not os.path.isfile(p):
        raise FileNotFoundError(f'ไม่พบภาพ: {name}')
    os.remove(p)


def migrate_flat_images(data_dir: str) -> dict:
    """ย้ายภาพที่วางแบน ๆ ใน data/ เข้าโฟลเดอร์ตามรหัสที่อ่านจากชื่อไฟล์

    ไฟล์ที่แกะรหัสไม่ได้จะถูกปล่อยไว้ที่เดิม ไม่เดาให้
    """
    moved, skipped = [], []
    idx = load_index(data_dir)
    for p in sorted(list_images(data_dir)):
        if os.path.dirname(os.path.abspath(p)) != os.path.abspath(data_dir):
            continue                                   # อยู่ในโฟลเดอร์สัตว์อยู่แล้ว
        parsed = parse_filename(p)
        if not parsed:
            skipped.append(os.path.basename(p))
            continue
        pid, name = parsed
        os.makedirs(patient_dir(data_dir, pid), exist_ok=True)
        dest = os.path.join(patient_dir(data_dir, pid), os.path.basename(p))
        if not os.path.exists(dest):
            shutil.move(p, dest)
        moved.append(os.path.relpath(dest, data_dir).replace(os.sep, '/'))
        if pid not in idx:
            idx[pid] = {'id': pid, 'name': name, 'note': '', 'group': '',
                        'created': date.today().isoformat()}
        elif not idx[pid].get('name') and name:
            idx[pid]['name'] = name
    save_index(data_dir, idx)
    return {'moved': moved, 'skipped': skipped}
