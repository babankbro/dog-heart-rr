# -*- coding: utf-8 -*-
"""สรุปช่วง RR ของสัตว์แต่ละตัวจากทุกภาพที่มี ด้วยหลายวิธีเพื่อเทียบกัน

ค่า RR ที่ได้จากภาพหนึ่งใบมีทั้งช่วงที่วัดจากจังหวะติดกันจริง และช่วงที่ยาวผิดปกติเพราะ
จังหวะกลางหาย หรือสั้นผิดปกติเพราะนับซ้ำ ค่ากลางแบบต่าง ๆ ทนต่อค่าผิดปกติไม่เท่ากัน
โมดูลนี้จึงคำนวณหลายวิธีพร้อมกันแล้วรายงานความไม่ลงรอยระหว่างวิธีไว้ด้วย
เพราะความไม่ลงรอยเองคือสัญญาณว่าข้อมูลของตัวนั้นมีค่าผิดปกติปนอยู่
"""
import math
import statistics
from typing import Dict, List, Optional, Sequence

MID_N = 20          # จำนวนค่าตรงกลางที่ใช้ในวิธีที่ 2
METHODS = ('mean_all', 'mid20', 'median')


def middle_values(values: Sequence[float], n: int = MID_N) -> List[float]:
    """ค่าตรงกลาง n ค่าหลังเรียงลำดับ — ตัดหางสองข้างเท่า ๆ กัน

    ถ้ามีน้อยกว่า n ค่า คืนทั้งหมด (ไม่เติมค่าปลอมและไม่ทิ้งข้อมูล)
    จำนวนที่ต้องตัดออกอาจเป็นเลขคี่ กรณีนั้นตัดข้างล่างมากกว่าหนึ่งค่า
    เพราะช่วงที่ยาวผิดปกติจากจังหวะที่หายอันตรายกว่าช่วงที่สั้น
    """
    s = sorted(values)
    if len(s) <= n:
        return s
    drop = len(s) - n
    lo = drop - drop // 2
    return s[lo:lo + n]


def describe(values: Sequence[float]) -> Optional[Dict[str, float]]:
    """สถิติพื้นฐานของชุดค่าหนึ่ง คืน None เมื่อไม่มีข้อมูล"""
    v = [float(x) for x in values]
    if not v:
        return None
    n = len(v)
    sd = statistics.stdev(v) if n > 1 else 0.0
    return {'n': n, 'mean': statistics.fmean(v), 'median': statistics.median(v),
            'min': min(v), 'max': max(v), 'sd': sd,
            'sem': sd / math.sqrt(n) if n else 0.0}


def summarize(values: Sequence[float]) -> Optional[Dict[str, object]]:
    """สรุปค่า RR ของสัตว์หนึ่งตัวด้วยสามวิธี พร้อมความคลาดเคลื่อนของแต่ละวิธี

    mean_all  ค่าเฉลี่ยของทุกช่วง — ใช้ข้อมูลครบแต่ถูกค่าผิดปกติลากง่ายที่สุด
    mid20     เรียงแล้วเอา 20 ค่าตรงกลางมาเฉลี่ย — ตัดหางทั้งสองข้างทิ้ง
    median    มัธยฐานของทุกช่วง — ทนค่าผิดปกติที่สุด แต่ทิ้งข้อมูลเชิงปริมาณไปมาก

    ความคลาดเคลื่อนรายงานเป็น sd (การกระจายของข้อมูล) และ sem (ความไม่แน่นอนของ
    ค่ากลางที่ประมาณได้) สำหรับมัธยฐานใช้ตัวประกอบ 1.2533 ตามค่าประมาณเชิงเส้นกำกับ
    ของความแปรปรวนของมัธยฐานเมื่อข้อมูลมาจากการแจกแจงปกติ
    """
    base = describe(values)
    if base is None:
        return None
    mid = describe(middle_values(values))
    out = {
        'n': base['n'], 'min': base['min'], 'max': base['max'],
        'sd': base['sd'], 'range': base['max'] - base['min'],
        'mean_all': {'value': base['mean'], 'n_used': base['n'],
                     'sd': base['sd'], 'sem': base['sem']},
        'mid20': {'value': mid['mean'], 'n_used': mid['n'],
                  'sd': mid['sd'], 'sem': mid['sem']},
        'median': {'value': base['median'], 'n_used': base['n'],
                   'sd': base['sd'],
                   'sem': 1.2533 * base['sem']},
    }
    vals = [out[m]['value'] for m in METHODS]
    # ความไม่ลงรอยระหว่างวิธี — สูงแปลว่ามีค่าผิดปกติปนจนวิธีที่ทนต่างกันให้คำตอบต่างกัน
    out['spread'] = max(vals) - min(vals)
    out['spread_pct'] = 100.0 * out['spread'] / out['median']['value'] if out['median']['value'] else 0.0
    return out
