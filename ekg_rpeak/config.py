"""ค่าตั้งทั้งหมดของไปป์ไลน์ รวมไว้ที่เดียว"""
from dataclasses import dataclass, replace
from typing import Optional


@dataclass
class Config:
    # ---------- โมเดล ----------
    crop_weights: str = 'models/crop_best.pt'
    point_weights: str = 'models/point_ink_best.pt'   # เทรนด้วยภาพ ink mask
    crop_conf: float = 0.40
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
    point_pre: str = 'ink'         # 'ink' | 'gray' | 'gray_contrast' | 'none'
                                   # ต้องตรงกับที่ใช้เทรน weights ที่โหลดอยู่

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
    anchor_expand: float = 0.05
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
    # โหมด train_match — สัดส่วนวัดจาก label ของ dataset 54 ภาพ
    # กว้างกว่านี้ conf เฉลี่ยขึ้นเล็กน้อย แต่ครอปข้างเคียงซ้อนกันจนรายงานยอด R ตัวเดียวกัน
    # แล้วถูกตัดเป็นจุดซ้ำ ทำให้จังหวะหายไปหนึ่ง — 1.07 เป็นค่าเดียวที่ได้ครบ 127/127
    train_frame_w_ratio: float = 1.07   # ความกว้างเฟรม = ค่านี้ x ระยะ RR
    train_frame_h_ratio: float = 1.25   # ความสูงเฟรม  = ค่านี้ x แอมพลิจูด R (จูนกับ weights ink)
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
