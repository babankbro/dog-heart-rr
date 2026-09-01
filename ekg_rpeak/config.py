"""ค่าตั้งทั้งหมดของไปป์ไลน์ รวมไว้ที่เดียว"""
from dataclasses import dataclass, replace
from typing import Optional


@dataclass
class Config:
    # ---------- โมเดล ----------
    crop_weights: str = 'models/crop_best.pt'
    point_weights: str = 'models/point_ink_best.pt'   # เทรนด้วยภาพ ink mask
    # 0.40 ทำให้จังหวะจริงหลุดไปหลายจังหวะต่อภาพ วัดกับชุดข้อมูล 67 ภาพแล้ว
    # ลดเหลือ 0.20 ได้กล่องเพิ่ม 76 ใบ (24 ภาพได้เพิ่ม ไม่มีภาพไหนได้น้อยลง)
    # โดยมีกล่องที่น่าสงสัยเพิ่มเพียง 13 ใบ และ flag รวมลดลง
    crop_conf: float = 0.20
    crop_iou: float = 0.35
    crop_imgsz: int = 512          # จาก args.yaml ของ run ที่เทรนจริง
    point_conf: float = 0.05    # โมเดลให้ conf ต่ำ (สูงสุด ~0.4) การกรองสูงกว่านี้ตัดจุดที่ถูกทิ้ง
    point_iou: float = 0.50
    point_imgsz: int = 512
    # โมเดลจุดเป็น multi-class landmark detector — คลาสของ weights ชุดนี้คือ
    # {0:P1, 1:P2, 2:P3, 3:Q1, 4:Q2, 5:R, 6:S1, 7:S2, 8:T1, 9:T2, 10:T3}
    # ถ้าเปลี่ยน weights ให้ตรวจด้วย `cli info` ก่อน (None = ไม่กรองคลาส ไม่แนะนำ)
    r_class_id: Optional[int] = 5

    # ---------- preprocessing ----------
    blackhat_ksize: int = 5        # ใช้กับโมเดลครอปเท่านั้น ต้องตรงกับสคริปต์เทรน
    blackhat_thr: int = 15
    # วิธี binarization ของภาพที่ป้อนโมเดลครอป — ดู crop_preprocess()
    # 'blackhat' คือวิธีเดียวกับชุดเทรน ค่าเริ่มต้นนี้ไม่ใช่ จึงเป็นการเปลี่ยนโดเมนที่โมเดลเห็น
    # เลือกไว้เพราะเส้นกริดหลัก 5 mm รอด tophat_gray มาเป็นแท่งทึบ ยอด R ที่พาดทับกริด
    # จึงกลืนหายไป (เช่น methung 1 กล่องที่ 11 กับ 13) การทำ tophat บนแชนเนลแดงลบกริด
    # ตั้งแต่ต้นทาง ผลจึงไม่ไวต่อ threshold — เก็บได้ครบทุกค่าตั้งแต่ thr 18 ถึง 45
    # ตัวเลขรวมของทั้งชุดอยู่ใน README หัวข้อ binarization
    crop_pre: str = 'tophat_red'   # ดูรายชื่อโหมดที่ CROP_PRE_MODES ใน preprocess.py
    # กระดาษ EKG พิมพ์กริดด้วยหมึกแดง ในแชนเนลแดงกริดจึงเกือบสว่างเท่ากระดาษ
    # ส่วนเส้นคลื่นสีดำยังเข้มอยู่ ใช้แชนเนลเดียวจึงแยกเส้นออกจากกริดได้โดยไม่ต้อง
    # พึ่ง saturation — ทนกับสแกนที่สีเพี้ยนหรือกริดจางกว่าปกติ
    # ปลายยอด R เขียนด้วยหมึกจาง threshold เดียวจึงตัดทิ้งเสมอ ไม่ว่าจะตั้งที่ไหน:
    # ตั้งเข้มก็เสียยอด ตั้งอ่อนก็ได้กระดาษกับกริดมาด้วย hysteresis แก้ตรงนี้ —
    # ใช้เกณฑ์เข้มหาแกนเส้นก่อน แล้วต่อยอดด้วยเกณฑ์อ่อนเฉพาะส่วนที่ติดกับแกนเท่านั้น
    crop_pre_hyst: float = 0.0   # ผ่อนเกณฑ์ไปทางกระดาษกี่ส่วน (0 = ไม่ใช้)
    crop_pre_close: int = 0      # เชื่อมเส้นที่ขาดเป็นช่วง ๆ (0 = ไม่ทำ)
    crop_pre_dilate: int = 0     # ทำให้เส้นหนาขึ้นหลัง binarize (0 = ไม่ทำ)
    crop_pre_ksize: int = 15     # tophat: แกนของ closing ที่ใช้ประมาณพื้นหลัง
    crop_pre_thr: int = 35       # tophat: เข้มกว่าพื้นหลังเท่าไรจึงนับเป็นเส้น
    crop_pre_block: int = 31     # adaptive: ขนาดหน้าต่าง
    crop_pre_c: int = 10         # adaptive: ค่าที่หักออกจากค่าเฉลี่ยในหน้าต่าง
    # ดูรายชื่อโหมดที่ POINT_PRE_MODES ใน preprocess.py
    # 'ink' คือวิธีเดียวกับชุดเทรน 'red_ink' ตัดกริดด้วยแชนเนลแดงแทน saturation
    # แล้วดึงคอนทราสต์ก่อน threshold — วัดกับ 67 ภาพแล้วโมเดลจุดยืนยันเพิ่มจาก
    # 89.1% เป็น 94.0% ของจุดทั้งหมด (พึ่ง anchor ลดจาก 170 เหลือ 101 จุด)
    point_pre: str = 'red_ink'

    # ---------- ink mask / anchor ----------
    ink_dark_v: int = 160          # เพดานความสว่างเริ่มต้นที่นับเป็นเส้นคลื่น
    ink_dark_v_max: int = 235      # เพดานสูงสุดตอนไล่ปรับอัตโนมัติ
    ink_sat_max: int = 70          # กริดมีสี saturation สูงกว่านี้จึงถูกตัดทิ้ง
    ink_drop_full_cols: bool = True
    ink_adaptive: bool = True
    ink_keep_trace_only: bool = True   # เก็บเฉพาะกลุ่มพิกเซลใหญ่ (เส้นคลื่น) ทิ้งจุดกริดที่รอดมา
    ink_min_area_frac: float = 0.15    # กลุ่มที่เล็กกว่านี้เทียบกับกลุ่มใหญ่สุด ถือว่าไม่ใช่เส้นคลื่น
    ink_min_run: int = 5               # ยอด R ต้องมาจากหมึกที่ต่อเนื่องกันอย่างน้อยกี่พิกเซลในแนวตั้ง
    ink_min_col_frac: float = 0.80
    ink_max_frac: float = 0.40
    anchor_expand: float = 0.05      # ขยาย ROI แนวนอนก่อนหายอด
    # ขยาย ROI แนวตั้งด้วย เพราะกล่องจากโมเดลครอปมัก "ตัดยอด" ทิ้ง:
    # โมเดลนั้นกินภาพ blackhat ที่ผ่าน threshold แล้ว ปลายยอด R ที่หมึกจาง
    # (เทา ~150-200) หลุดจาก binarization ไป ขอบบนของกล่องจึงต่ำกว่ายอดจริง
    # หลายสิบพิกเซล ถ้าไม่ขยาย anchor จะไปติดขอบกล่องแทนที่จะเจอยอด
    anchor_expand_y: float = 0.35
    anchor_min_col_frac: float = 0.50
    anchor_on_rpeak: bool = True

    # ตัดกล่องหัว/ท้ายแถวที่ไม่ใช่คลื่น เช่น calibration pulse หรือเศษคลื่นที่ขอบภาพ
    drop_edge_non_beats: bool = True
    edge_amp_ratio: float = 0.4      # ยอดเตี้ยกว่านี้เทียบกับจังหวะกลางแถว = ไม่ใช่จังหวะ
    edge_width_ratio: float = 3.0    # ยอดกว้างกว่านี้ = ยอดแบน ไม่ใช่ยอดแหลมของ R
    edge_min_box_ratio: float = 0.5  # กล่องแคบกว่านี้เทียบกับกล่องกลางแถว = เศษที่ขอบภาพ
    edge_peak_margin: float = 0.15   # ยอดอยู่ในขอบซ้าย/ขวาของกล่องเกินสัดส่วนนี้ = ไม่ใช่จังหวะ

    # ---------- เรขาคณิตของครอป ----------
    out_size: int = 512
    crop_mode: str = 'train_match'  # 'train_match' | 'mm' | 'anchored' | 'height' | 'pitch' | 'box' | 'stretch'
    crop_side_ratio: float = 1.3
    pad_ratio: float = 0.15
    pad_mode: str = 'replicate'    # 'replicate' | 'white'
    shift_inside: bool = False     # เลื่อนกรอบเข้ามาในภาพแทนการเติมขอบ ถ้ายังพอเลื่อนได้
    train_px_per_mm: float = 18.0    # ครอปครอบคลุม 512/18 = 28.4 mm — จูนจากภาพจริง 6 ภาพ
    train_px_per_mm_y: float = None  # None = จัตุรัส; ตั้งค่าเพื่อยืดแนวตั้งเหมือน dataset
    train_anchor_xfrac: float = 0.454   # วัดจาก label ของ dataset (n=54)
    train_anchor_yfrac: float = 0.095
    anchored_pad: float = 0.15          # โหมด anchored: ขยายกล่องกี่ส่วนก่อนยืดเป็นจัตุรัส
    # โหมด train_match — สัดส่วนเดียวกับสคริปต์ที่เทรนโมเดลจุด (FRAME_W/H_RATIO)
    # เทียบครอปของเรากับครอปชุดเทรนด้วย compare_domain แล้ว ค่านี้ให้รูปทรงใกล้ที่สุด
    # (ระยะห่างของโปรไฟล์หมึก 0.21 เทียบกับ 0.50 ของ 1.07/1.25) เห็น P กับ T ครบเฟรม
    # ราคาที่จ่าย: ครอปข้างเคียงซ้อนกันมากขึ้น จุดซ้ำเพิ่มจาก 15 เป็น 18 และโมเดลจุด
    # ยืนยันน้อยลง 10 จุดจาก 1380 — แลกกับ flag ที่ลดจาก 83 เหลือ 80
    train_frame_w_ratio: float = 1.40   # ความกว้างเฟรม = ค่านี้ x ระยะ RR
    train_frame_h_ratio: float = 1.49   # ความสูงเฟรม  = ค่านี้ x แอมพลิจูด R
    center_sigma: float = 0.30

    # ---------- รวมผล ----------
    row_tol_ratio: float = 0.6
    dedup_ratio: float = 0.5
    # เกณฑ์ flag เทียบกับช่วงข้างเคียง ไม่ใช่มัธยฐานทั้งภาพ
    flag_window: int = 4             # ดูข้างละกี่ช่วงเป็นฐานเทียบ
    flag_high_ratio: float = 1.5     # สูงกว่านี้ = อาจมีจังหวะหาย
    flag_low_ratio: float = 0.5      # ต่ำกว่านี้ = อาจเป็นจุดซ้ำ
    landmark_dedup_ratio: float = 0.25   # ครอปที่ซ้อนกันรายงาน landmark เดียวกันซ้ำ
    point_mode: str = 'refine'     # 'refine' | 'model_only' | 'anchor_only'
    max_refine_ratio: float = 0.35
    trust_model_conf: float = 0.30
    anchor_conf: float = 0.30
    batch: int = 16

    # ---------- ระบบพิกัดอ้างอิงจากเส้นกริด ----------
    # ที่มาของ px ต่อ mm
    # 'minor' วัดจากคาบกริดเล็ก 1 mm ด้วย autocorrelation — เร็วแต่พลาดเมื่อกริดเล็กจาง
    #         จนไปล็อกกับพหุคูณของคาบจริง ทำให้ค่าผิดเป็นจำนวนเท่า
    # 'major' ถือว่าหนึ่งช่องกริดหลักเท่ากับ grid_mm มิลลิเมตร แล้วตัดสินว่าคาบไหนคือ
    #         ช่องหลักด้วยช่วงอัตราการเต้นที่เป็นไปได้
    # 'auto'  ใช้ 'minor' ก่อน ถ้าให้อัตราการเต้นที่เป็นไปไม่ได้จึงเปลี่ยนไปใช้ 'major'
    scale_source: str = 'auto'   # 'auto' | 'minor' | 'major'
    # ช่วงอัตราการเต้นที่ใช้ตัดสินว่าสเกลที่วัดได้เป็นไปได้ไหม
    # เป็นความรู้ล่วงหน้าทางสรีรวิทยา ไม่ใช่การวัด จึงต้องรายงานให้ผู้ใช้เห็นเสมอ
    # ต้องแคบพอจะจับความผิดพลาดแบบผิดเป็นจำนวนเท่าได้ ค่า 20-400 เดิมกว้างเกินไป
    scale_hr_lo: float = 40.0
    scale_hr_hi: float = 300.0
    grid_mm: float = 5.0             # ระยะห่างของเส้นกริดหลัก (มาตรฐานกระดาษ EKG)
    grid_refine: bool = True         # ปรับระยะด้วย least squares จากตำแหน่งเส้นจริง
    grid_from_lines: bool = True     # ใช้เส้นกริดหลักเป็นตัววัด px ต่อ mm (แม่นกว่ากริดเล็ก)
    grid_origin_mode: str = 'before_first_beat'   # 'before_first_beat' | 'nearest'

    # ---------- สเกลเวลา ----------
    px_per_mm: Optional[float] = None
    auto_px_per_mm: bool = True
    paper_speed_mm_s: float = 25.0

    def with_(self, **kw) -> 'Config':
        """คืน Config ใหม่ที่แก้บางค่า โดยไม่แก้ของเดิม"""
        return replace(self, **kw)
