"""ดึงภาพจากโครงโฟลเดอร์ภายนอกเข้าทะเบียนสัตว์

โครงที่รองรับ: <ราก>/<ประเภท>/<รหัส ชื่อ>/<crop>/*.jpg
ชื่อโฟลเดอร์ประเภทถูกใช้เป็น group ของสัตว์ทุกตัวที่อยู่ข้างใน

ต้นทางเป็นข้อมูลที่คนจัดเอง จึงไม่สะอาดเสมอ ตัวนี้จึงยอมรับความเพี้ยนสามอย่าง
ที่เจอจริงและแก้ให้เอง แทนที่จะล้มทั้งงาน
  - โฟลเดอร์ครอปสะกดไม่ตรงกัน (crop / Crop)
  - ไฟล์ภาพไม่มีนามสกุล หรือมีนามสกุลที่ไม่ใช่ภาพ (ตัดสินจากไบต์แรกของไฟล์)
  - ชื่อไฟล์ยาวและอ่านไม่รู้เรื่อง จึงตั้งชื่อใหม่เป็น "<รหัส> <ชื่อ> <ลำดับ>.jpg"
  - รหัสเดิมในทะเบียนสะกดต่างแค่ตัวพิมพ์ (C56153 vs c56153) ซึ่งบน Windows
    เป็นโฟลเดอร์เดียวกัน แต่ในทะเบียนเป็นคนละระเบียน ทำให้เห็นสัตว์ซ้ำสองรายการ
"""
import os
import re
import shutil
from datetime import date
from typing import Dict, List, Optional

from . import patients as pt

# ไบต์แรกของแต่ละชนิดภาพ — ชื่อไฟล์ในชุดข้อมูลจริงเชื่อไม่ได้ บางไฟล์ไม่มีนามสกุลเลย
MAGIC = {b'\xff\xd8\xff': '.jpg', b'\x89PNG': '.png'}
GROUP_PREFIX_RE = re.compile(r'^\s*\d+\s*[.)]?\s*')     # "1. Normal" -> "Normal"


def image_ext(path: str) -> Optional[str]:
    """นามสกุลที่ควรใช้กับไฟล์นี้ ตัดสินจากเนื้อไฟล์ คืน None ถ้าไม่ใช่ภาพ"""
    try:
        with open(path, 'rb') as f:
            head = f.read(8)
    except OSError:
        return None
    for magic, ext in MAGIC.items():
        if head.startswith(magic):
            return ext
    return None


def clean_group(folder: str) -> str:
    """ตัดเลขลำดับหน้าชื่อโฟลเดอร์ออก เหลือชื่อประเภทล้วน"""
    return GROUP_PREFIX_RE.sub('', folder).strip()


def split_folder_name(folder: str):
    """"<รหัส> <ชื่อ>" -> (รหัส, ชื่อ) ชื่ออาจว่างได้ คืน None ถ้ารหัสใช้ไม่ได้"""
    parts = folder.strip().split(None, 1)
    if not parts or not pt.valid_id(parts[0]):
        return None
    return parts[0], (parts[1].strip() if len(parts) > 1 else '')


def find_subdir(parent: str, keyword: str) -> Optional[str]:
    """โฟลเดอร์ย่อยที่ชื่อมี keyword โดยไม่สนตัวพิมพ์ (crop / Crop)"""
    for name in sorted(os.listdir(parent)):
        if keyword in name.lower() and os.path.isdir(os.path.join(parent, name)):
            return os.path.join(parent, name)
    return None


def import_tree(src: str, data_dir: str, subdir: str = 'crop',
                replace: bool = False, limit: int = 0) -> Dict[str, list]:
    """คัดลอกภาพจาก src เข้า data_dir แล้วปรับทะเบียนให้ตรง

    replace=True ลบภาพเดิมของสัตว์ที่ชื่อชนกันก่อนใส่ชุดใหม่
    limit จำกัดจำนวนภาพต่อตัว (0 = เอาทั้งหมด)
    """
    if not os.path.isdir(src):
        raise FileNotFoundError(f'ไม่พบโฟลเดอร์ต้นทาง: {src}')
    idx = pt.load_index(data_dir)
    added: List[dict] = []
    skipped: List[dict] = []

    for group_folder in sorted(os.listdir(src)):
        group_path = os.path.join(src, group_folder)
        if not os.path.isdir(group_path):
            continue
        group = clean_group(group_folder)

        for dog_folder in sorted(os.listdir(group_path)):
            dog_path = os.path.join(group_path, dog_folder)
            if not os.path.isdir(dog_path):
                continue
            parsed = split_folder_name(dog_folder)
            if parsed is None:
                skipped.append({'folder': f'{group_folder}/{dog_folder}',
                                'reason': 'แกะรหัสจากชื่อโฟลเดอร์ไม่ได้'})
                continue
            pid, name = parsed

            crop_dir = find_subdir(dog_path, subdir)
            if crop_dir is None:
                skipped.append({'folder': f'{group_folder}/{dog_folder}',
                                'reason': f'ไม่มีโฟลเดอร์ที่ชื่อมี "{subdir}"'})
                continue
            srcs = [(f, image_ext(os.path.join(crop_dir, f)))
                    for f in sorted(os.listdir(crop_dir))
                    if os.path.isfile(os.path.join(crop_dir, f))]
            srcs = [(f, e) for f, e in srcs if e]
            if not srcs:
                skipped.append({'folder': f'{group_folder}/{dog_folder}',
                                'reason': 'ไม่มีไฟล์ภาพในโฟลเดอร์ครอป'})
                continue
            if limit:
                srcs = srcs[:limit]

            # รหัสเดิมที่ต่างกันแค่ตัวพิมพ์คือตัวเดียวกัน เก็บไว้ทั้งคู่จะเห็นซ้ำในหน้าเว็บ
            for old in [k for k in idx if k != pid and k.lower() == pid.lower()]:
                idx.pop(old)

            dest_dir = pt.patient_dir(data_dir, pid)
            if replace and os.path.isdir(dest_dir):
                shutil.rmtree(dest_dir)
            os.makedirs(dest_dir, exist_ok=True)

            # ตั้งชื่อใหม่ให้อ่านออกและมีนามสกุลเสมอ ชื่อเดิมบางไฟล์เป็น "ไฟล์ - <เวลา>Z"
            stem = f'{pid} {name}'.strip()
            files = []
            for i, (fname, ext) in enumerate(srcs, 1):
                dest = os.path.join(dest_dir, f'{stem} {i}{ext}')
                shutil.copy2(os.path.join(crop_dir, fname), dest)
                files.append(os.path.basename(dest))

            rec = idx.get(pid, {'id': pid, 'created': date.today().isoformat()})
            idx[pid] = {**pt.FIELD_DEFAULTS, **rec, 'id': pid,
                        'name': name or rec.get('name', ''), 'group': group}
            added.append({'id': pid, 'name': name, 'group': group, 'images': files})

    pt.save_index(data_dir, idx)
    return {'added': added, 'skipped': skipped,
            'n_patients': len(added), 'n_images': sum(len(a['images']) for a in added)}
