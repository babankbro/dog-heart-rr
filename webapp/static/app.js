const $ = (id) => document.getElementById(id);
const OPTS = ['point_mode', 'crop_mode', 'point_pre', 'r_class_id',
              'px_per_mm', 'crop_conf', 'point_conf'];

let patients = [];       // ทะเบียนสัตว์ทั้งหมด
let currentPid = null;   // รหัสสัตว์ที่เลือกอยู่
let current = null;      // ชื่อภาพที่รันล่าสุด
let currentRev = 0;      // เลขรุ่นของผล ใช้ทำ URL ที่เบราว์เซอร์แคชได้
let lastData = null;     // ผลตรวจจับของภาพเดี่ยว
let patientData = null;  // ผลวิเคราะห์ทั้งตัว

async function api(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* ไม่ใช่ json */ }
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

const jsonPost = (url, body, method = 'POST') => api(url, {
  method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

function overrides() {
  const o = {};
  for (const k of OPTS) {
    const v = $(k).value;
    if (v !== '' && v !== null) o[k] = v;
  }
  return o;
}

function setErr(msg) { $('err').textContent = msg || ''; }

function chip(label, value, cls = '') {
  return `<div class="chip ${cls}"><span>${label}</span> <b>${value}</b></div>`;
}

// ---------------------------------------------------------------- ทะเบียนสัตว์

async function loadHealth() {
  const h = await api('/api/health');
  const mark = (ok, label) => `<span class="${ok ? 'ok' : 'bad'}">${ok ? '●' : '○'} ${label}</span>`;
  $('health').innerHTML = mark(h.crop_weights, 'crop weights') + ' &nbsp; ' +
                          mark(h.point_weights, 'point weights');
}

async function loadPatients(keepPid) {
  patients = await api('/api/patients');
  const sel = $('patientSelect');
  const want = keepPid || currentPid || (patients[0] && patients[0].id);
  sel.innerHTML = '';
  for (const p of patients) {
    const o = document.createElement('option');
    o.value = p.id;
    o.textContent = `${p.id}${p.name ? '  ' + p.name : ''}  (${p.n_images})`;
    sel.appendChild(o);
  }
  if (!patients.length) {
    const o = document.createElement('option');
    o.textContent = 'ยังไม่มีสัตว์ — กด "เพิ่มตัวใหม่"';
    o.disabled = true;
    sel.appendChild(o);
    currentPid = null;
    fillImages([]);
    $('patientHead').innerHTML = '';
    $('patientAgg').innerHTML = '';
    $('patientGrid').innerHTML = '';
    return;
  }
  sel.value = patients.some(p => p.id === want) ? want : patients[0].id;
  selectPatient(sel.value);
}

function patientById(pid) { return patients.find(p => p.id === pid); }

function fillImages(list) {
  const sel = $('imageSelect');
  sel.innerHTML = '';
  for (const name of list) {
    const o = document.createElement('option');
    o.value = name;
    o.textContent = name.split('/').pop();
    sel.appendChild(o);
  }
  $('imgCount').textContent = list.length ? `${list.length} ภาพ` : 'ยังไม่มีภาพ';
  if (list.length) sel.value = list[0];
}

function clearImageView() {
  current = null; currentRev = 0; lastData = null;
  setViewing(null);
  for (const id of ('overlay crops mask').split(' ')) $(id).removeAttribute('src');
  for (const id of ('stats warn table rrSummary rrChart rrTable').split(' ')) $(id).innerHTML = '';
  $('warn').hidden = true;
  $('csvLink').removeAttribute('href');
}


function selectPatient(pid) {
  currentPid = pid;
  const p = patientById(pid);
  fillImages(p ? p.images : []);
  clearImageView();          // ภาพของตัวก่อนหน้าต้องหายไปทันทีที่เปลี่ยนตัว
  patientData = null;
  renderPatientHead(p);
  $('patientAgg').innerHTML = '';
  $('patientGrid').innerHTML = p && p.images.length
    ? '<p class="hint">กด "วิเคราะห์ทั้งตัว" เพื่อประมวลผลทุกภาพของสัตว์ตัวนี้</p>'
    : '<p class="hint">ยังไม่มีภาพ — กด "+ เพิ่มภาพ" เพื่ออัปโหลดภาพแรก</p>';
}

function renderPatientHead(p) {
  if (!p) { $('patientHead').innerHTML = ''; return; }
  $('patientHead').innerHTML =
    `<h3>${p.name || '(ยังไม่ตั้งชื่อ)'} <span class="pid">${p.id}</span></h3>` +
    `<div class="muted">${p.note || ''}${p.created ? '  ·  เพิ่มเมื่อ ' + p.created : ''}</div>`;
}

// ---------------------------------------------------------------- หน้ารวมของสัตว์

function renderPatient(d) {
  patientData = d;
  renderPatientHead({ ...d.patient, images: [] });
  const a = d.aggregate;
  const hr = a.hr_mean ? `${a.hr_mean.toFixed(0)} bpm` : '—';
  const hrRange = (a.hr_min && a.hr_max) ? `${a.hr_min.toFixed(0)}–${a.hr_max.toFixed(0)}` : '—';
  $('patientAgg').innerHTML = [
    chip('ภาพ', a.n_images),
    chip('จังหวะรวม', a.beats),
    chip('จุด R รวม', a.r_peaks),
    chip('จากโมเดล', a.model, a.model ? 'good' : ''),
    chip('จาก anchor', a.anchor),
    chip('HR เฉลี่ยทุกภาพ', hr, 'good'),
    chip('ช่วง HR', hrRange),
    chip('RR เฉลี่ย', a.rr_mean_mm ? `${a.rr_mean_mm.toFixed(2)} mm` : '—'),
    chip('ต่างกันสูงสุดระหว่างภาพ', a.rr_spread_mm ? `${a.rr_spread_mm.toFixed(2)} mm` : '—',
         a.rr_spread_mm > 3 ? 'bad' : ''),
    chip('flag รวม', a.flags, a.flags ? 'bad' : ''),
  ].join('');

  const cards = d.images.map(im => {
    const hrTxt = im.hr ? `${im.hr.toFixed(0)} bpm` : '—';
    const rrTxt = im.rr_mean_mm ? `${im.rr_mean_mm.toFixed(2)} ± ${(im.rr_sd_mm || 0).toFixed(2)} mm` : '—';
    const warn = im.flags ? `<span class="tag bad">flag ${im.flags}</span>` : '';
    const anch = im.anchor ? `<span class="tag">anchor ${im.anchor}</span>` : '';
    return `<article class="pcard">
      <div class="pcard-img"><img alt=""
           src="/api/overlay?image=${encodeURIComponent(im.image)}&width=560&rev=${im.rev}"></div>
      <div class="pcard-body">
        <div class="pcard-title">${im.image.split('/').pop()}</div>
        <div class="pcard-stats">
          <span>จังหวะ <b>${im.beats}</b></span>
          <span>R <b>${im.r_peaks}</b></span>
          <span>HR <b>${hrTxt}</b></span>
          <span>RR <b>${rrTxt}</b></span>
          <span>px/mm <b>${im.px_per_mm ? im.px_per_mm.toFixed(2) : '—'}</b></span>
          ${anch}${warn}
        </div>
        <div class="row">
          <button class="btn tiny open-img" data-img="${im.image}">เปิดภาพนี้</button>
          <button class="btn tiny ghost danger del-img" data-img="${im.image}">ลบ</button>
        </div>
      </div>
    </article>`;
  }).join('');

  const errs = d.errors.map(e => `<div class="warn">⚠ ${e.image}: ${e.error}</div>`).join('');
  $('patientGrid').innerHTML = errs + (cards || '<p class="hint">ไม่มีภาพ</p>');

  $('patientGrid').querySelectorAll('.open-img').forEach(b => {
    b.onclick = () => { $('imageSelect').value = b.dataset.img; openTab('overlay'); run(); };
    // run() จะได้ผลเดิมจากแคชทันที เพราะ analyze รันไปแล้วด้วยค่าตั้งชุดเดียวกัน
  });
  $('patientGrid').querySelectorAll('.del-img').forEach(b => {
    b.onclick = () => deleteImage(b.dataset.img);
  });
}

async function analyzePatient() {
  if (!currentPid) { setErr('เลือกสัตว์ก่อน'); return; }
  const p = patientById(currentPid);
  if (!p || !p.images.length) { setErr('สัตว์ตัวนี้ยังไม่มีภาพ'); return; }
  setErr('');
  $('analyzeBtn').disabled = true;
  $('analyzeBtn').textContent = `กำลังวิเคราะห์ ${p.images.length} ภาพ...`;
  try {
    const d = await jsonPost(`/api/patients/${encodeURIComponent(currentPid)}/analyze`,
                             { overrides: overrides() });
    renderPatient(d);
    openTab('patient');
    if (d.images.length) {            // โหลดภาพแรกไว้ให้แท็บอื่นพร้อมใช้ทันที
      $('imageSelect').value = d.images[0].image;
      run();
    }
  } catch (e) {
    setErr(e.message);
  } finally {
    $('analyzeBtn').disabled = false;
    $('analyzeBtn').textContent = 'วิเคราะห์ทั้งตัว';
  }
}

// ---------------------------------------------------------------- ภาพเดี่ยว

function renderStats(d) {
  const s = d.stats;
  const hr = d.median_hr ? d.median_hr.toFixed(0) + ' bpm' : '—';
  const ppm = s.px_per_mm ? s.px_per_mm.toFixed(2) : '—';
  const flagged = d.rows.filter(r => r.flag).length;
  $('stats').innerHTML = [
    chip('จังหวะที่พบ', s.n_boxes),
    chip('จุด R', s.n_peaks),
    chip('จากโมเดล', s.n_model, s.n_model ? 'good' : ''),
    chip('จาก anchor', s.n_anchor),
    chip('ถูกปฏิเสธ', s.n_reject, s.n_reject ? 'bad' : ''),
    chip('ตัดซ้ำ', s.n_dup),
    chip('ตัดกล่องขอบ', s.n_edge_dropped ?? 0),
    chip('px/mm', ppm, d.scale_ok ? 'good' : 'bad'),
    chip('HR มัธยฐาน', hr),
    chip('แถวที่ติด flag', flagged, flagged ? 'bad' : ''),
    chip('landmark', s.n_landmarks ?? 0),
    d.grid ? chip('กริด 5 mm', `${d.grid.spacing_px} px`, 'good') : chip('กริด', 'ไม่พบ', 'bad'),
    d.grid && d.grid.resid_rms_px != null
      ? chip('กริดคลาดเคลื่อน', `±${d.grid.resid_rms_px} px, ไหล ${d.grid.drift_px >= 0 ? '+' : ''}${d.grid.drift_px} px`,
             Math.abs(d.grid.drift_px) < 0.5 ? 'good' : 'bad') : '',
  ].join('');

  const notes = [];
  if (!d.has_point_model)
    notes.push('ไม่พบ weights ของโมเดลจุด — ผลทั้งหมดมาจาก anchor (image processing)');
  if (d.has_point_model && !$('r_class_id').value)
    notes.push('ยังไม่ได้ตั้งคลาส R — โมเดลจุดตรวจ landmark หลายชนิด (P/Q/R/S/T) อาจได้จุดที่ไม่ใช่ R');
  if (!d.scale_ok && d.hr_from_pitch)
    notes.push(`px/mm อาจเพี้ยน: คำนวณได้ HR ~${d.hr_from_pitch.toFixed(0)} bpm ซึ่งอยู่นอกช่วงที่เป็นไปได้`);
  $('warn').hidden = notes.length === 0;
  $('warn').innerHTML = notes.map(n => `⚠ ${n}`).join('<br>');
}

function renderRR(d) {
  const rows = d.rows.filter(r => r.rr_mm !== '' && r.rr_mm !== null);
  const rr = d.rr;
  if (!rr || !rr.mean_mm) {
    $('rrSummary').innerHTML = chip('RR', 'คำนวณไม่ได้ (ไม่พบสเกลจากกริด)', 'bad');
    $('rrChart').innerHTML = ''; $('rrTable').innerHTML = '';
    return;
  }
  const gridMm = d.grid ? d.grid.mm : 5;
  $('rrSummary').innerHTML = [
    chip('จำนวน R', rr.n),
    chip('RR เฉลี่ย', `${rr.mean_mm.toFixed(2)} mm`, 'good'),
    chip('ส่วนเบี่ยงเบน', `± ${rr.sd_mm.toFixed(2)} mm`),
    chip('มัธยฐาน', `${rr.median_mm.toFixed(2)} mm`),
    chip('ต่ำสุด–สูงสุด', `${rr.min_mm.toFixed(2)} – ${rr.max_mm.toFixed(2)} mm`),
    chip('คิดเป็นช่องกริด', `${(rr.mean_mm / gridMm).toFixed(2)} ช่อง`),
    chip('เวลา', `${rr.mean_sec.toFixed(3)} s`),
    chip('อัตราการเต้น', `${rr.mean_bpm.toFixed(0)} bpm`),
  ].join('');

  const vals = rows.map(r => Number(r.rr_mm));
  const W = Math.max(320, vals.length * 26 + 60), H = 150, pad = 24;
  const maxV = Math.max(...vals, rr.mean_mm) * 1.15;
  const bw = (W - pad - 10) / vals.length;
  const y = v => H - pad - (v / maxV) * (H - pad - 10);
  const bars = vals.map((v, i) => {
    const out = Math.abs(v - rr.mean_mm) > 2 * rr.sd_mm && rr.sd_mm > 0;
    return `<rect class="bar${out ? ' out' : ''}" x="${pad + i * bw + 1}" y="${y(v)}" ` +
           `width="${Math.max(2, bw - 2)}" height="${H - pad - y(v)}"><title>ช่วงที่ ${i + 1}: ` +
           `${v.toFixed(2)} mm</title></rect>`;
  }).join('');
  $('rrChart').innerHTML =
    `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">` +
    `<line class="mean" x1="${pad}" x2="${W - 10}" y1="${y(rr.mean_mm)}" y2="${y(rr.mean_mm)}"/>` +
    `<text x="${W - 10}" y="${y(rr.mean_mm) - 4}" text-anchor="end">เฉลี่ย ${rr.mean_mm.toFixed(2)} mm</text>` +
    bars +
    `<line x1="${pad}" x2="${W - 10}" y1="${H - pad}" y2="${H - pad}" stroke="#39415280"/>` +
    `<text x="4" y="${y(maxV / 1.15) + 4}">${(maxV / 1.15).toFixed(1)}</text>` +
    `<text x="4" y="${H - pad + 4}">0</text>` +
    `<text x="${pad}" y="${H - 6}">ช่วงระหว่างจังหวะ (mm)</text></svg>`;

  const head = ['จังหวะ', 'x (mm)', 'RR (mm)', 'ช่องกริด', 'RR (s)', 'bpm', 'ที่มา', 'flag'];
  $('rrTable').innerHTML =
    '<tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr>' +
    d.rows.map(r => {
      const boxes = r.rr_mm === '' ? '' : (Number(r.rr_mm) / gridMm).toFixed(2);
      return `<tr class="${r.flag ? 'flagged' : ''}"><td>${r.r_index}</td><td>${r.x_mm}</td>` +
             `<td>${r.rr_mm}</td><td>${boxes}</td><td>${r.rr_sec}</td><td>${r.bpm}</td>` +
             `<td>${r.src}</td><td>${r.flag}</td></tr>`;
    }).join('');
}

function renderTable(d) {
  const only = $('onlyFlagged').checked;
  const rows = only ? d.rows.filter(r => r.flag) : d.rows;
  const head = '<tr>' + d.fields.map(f => `<th>${f}</th>`).join('') + '</tr>';
  const body = rows.map(r => {
    const cls = [r.flag ? 'flagged' : '', r.src === 'anchor' ? 'anchor' : ''].join(' ').trim();
    return `<tr class="${cls}">` + d.fields.map(f => `<td>${r[f] ?? ''}</td>`).join('') + '</tr>';
  }).join('');
  $('table').innerHTML = head + body;
}

function refreshOverlay() {
  if (!current) return;
  const q = new URLSearchParams({
    image: current,
    boxes: $('showBoxes').checked ? 1 : 0,
    marks: $('showMarks').checked ? 1 : 0,
    landmarks: $('showLandmarks').checked ? 1 : 0,
    origin: $('showOrigin').checked ? 1 : 0,
    grid: $('showGrid').checked ? 1 : 0,
    rev: currentRev,
  });
  const img = $('overlay');
  img.src = '/api/overlay?' + q;
  img.style.width = $('zoom').value + 'px';
  img.style.maxWidth = 'none';
}

const refreshCrops = () => {
  if (current) $('crops').src =
    `/api/crops?image=${encodeURIComponent(current)}&n=8&rev=${currentRev}`;
};
const refreshMask = () => {
  if (!current) return;
  const i = Math.max(0, parseInt($('maskIndex').value || '0', 10));
  $('mask').src = `/api/mask?image=${encodeURIComponent(current)}&index=${i}&rev=${currentRev}`;
};

function setViewing(name, busy) {
  const el = $('viewing');
  if (!name) { el.hidden = true; el.innerHTML = ''; return; }
  el.hidden = false;
  el.innerHTML = `กำลังดู <b>${name.split('/').pop()}</b>` +
                 (busy ? ' <span class="spin">กำลังโหลด...</span>' : '');
}


async function run() {
  const name = $('imageSelect').value;
  if (!name) { setErr('เลือกภาพก่อน'); return; }
  setErr('');
  setViewing(name, true);
  $('runBtn').disabled = true;
  $('runBtn').textContent = 'กำลังตรวจจับ...';
  try {
    const d = await jsonPost('/api/detect', { image: name, overrides: overrides() });
    current = name;
    currentRev = d.rev;
    lastData = d;
    renderStats(d); renderTable(d); renderRR(d);
    $('csvLink').href = '/api/csv?image=' + encodeURIComponent(name);
    $('maskIndex').max = Math.max(0, d.stats.n_boxes - 1);
    refreshOverlay(); refreshCrops(); refreshMask();
    setViewing(name, false);
  } catch (e) {
    setErr(e.message);
    setViewing(null);
  } finally {
    $('runBtn').disabled = false;
    $('runBtn').textContent = 'ตรวจจับเฉพาะภาพนี้';
  }
}

// ---------------------------------------------------------------- จัดการภาพ

async function uploadImages(files) {
  if (!currentPid) { setErr('เลือกหรือสร้างสัตว์ก่อนอัปโหลด'); return; }
  const fd = new FormData();
  files.forEach(f => fd.append('files', f));
  setErr('');
  try {
    const r = await api(`/api/patients/${encodeURIComponent(currentPid)}/images`,
                        { method: 'POST', body: fd });
    await loadPatients(currentPid);
    if (r.saved.length) $('imageSelect').value = r.saved[0];
    if (r.failed.length) setErr(r.failed.map(f => `${f.file}: ${f.reason}`).join('\n'));
    await analyzePatient();
  } catch (e) {
    setErr(e.message);
  }
}

async function deleteImage(name) {
  if (!confirm(`ลบภาพ ${name.split('/').pop()} ?`)) return;
  try {
    await api(`/api/patients/${encodeURIComponent(currentPid)}/images?name=` +
              encodeURIComponent(name.split('/').pop()), { method: 'DELETE' });
    if (current === name) { current = null; lastData = null; }
    await loadPatients(currentPid);
    const p = patientById(currentPid);
    if (p && p.images.length) await analyzePatient();
  } catch (e) {
    setErr(e.message);
  }
}

// ---------------------------------------------------------------- events

$('patientSelect').onchange = () => selectPatient($('patientSelect').value);
$('imageSelect').onchange = run;      // คลิกเลือกภาพ ทุกแท็บเปลี่ยนตามทันที
$('imageSelect').ondblclick = run;
$('runBtn').onclick = run;
$('analyzeBtn').onclick = analyzePatient;

$('newPatientBtn').onclick = () => {
  $('newPatientForm').hidden = false;
  $('npId').value = ''; $('npName').value = ''; $('npNote').value = '';
  $('npId').focus();
};
$('npCancel').onclick = () => { $('newPatientForm').hidden = true; };

$('newPatientForm').onsubmit = async (e) => {
  e.preventDefault();
  setErr('');
  try {
    const p = await jsonPost('/api/patients', {
      id: $('npId').value.trim(), name: $('npName').value.trim(), note: $('npNote').value.trim(),
    });
    $('newPatientForm').hidden = true;
    await loadPatients(p.id);
    $('uploader').click();          // ต่อด้วยการอัปโหลดภาพแรกทันที
  } catch (err) {
    setErr(err.message);
  }
};

$('editPatientBtn').onclick = async () => {
  const p = patientById(currentPid);
  if (!p) return;
  const name = prompt('ชื่อสัตว์', p.name || '');
  if (name === null) return;
  const note = prompt('หมายเหตุ', p.note || '');
  try {
    await jsonPost(`/api/patients/${encodeURIComponent(p.id)}`,
                   { name, note: note === null ? p.note : note }, 'PATCH');
    await loadPatients(p.id);
  } catch (e) { setErr(e.message); }
};

$('delPatientBtn').onclick = async () => {
  const p = patientById(currentPid);
  if (!p) return;
  if (!confirm(`ลบ ${p.id} ${p.name || ''} ออกจากทะเบียน?`)) return;
  const withImages = p.n_images > 0 &&
    confirm(`ลบภาพทั้ง ${p.n_images} ใบด้วยหรือไม่?\n\nOK = ลบภาพด้วย   Cancel = เก็บภาพไว้`);
  try {
    await api(`/api/patients/${encodeURIComponent(p.id)}?with_images=${withImages ? 1 : 0}`,
              { method: 'DELETE' });
    currentPid = null;
    await loadPatients();
  } catch (e) { setErr(e.message); }
};

$('delImageBtn').onclick = () => {
  const name = $('imageSelect').value;
  if (name) deleteImage(name);
};

$('uploader').onchange = async (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  if (files.length) await uploadImages(files);
};

$('migrateBtn').onclick = async () => {
  try {
    const r = await api('/api/migrate', { method: 'POST' });
    await loadPatients();
    setErr(r.moved.length
      ? `จัดกลุ่มแล้ว ${r.moved.length} ภาพ` +
        (r.skipped.length ? ` (ข้าม ${r.skipped.length} ที่แกะรหัสไม่ได้)` : '')
      : 'ไม่มีภาพที่ต้องจัดกลุ่ม');
  } catch (e) { setErr(e.message); }
};

$('resetOptsBtn').onclick = async () => {
  for (const k of OPTS) if ($(k).tagName !== 'SELECT') $(k).value = '';
  await loadDefaults();
  setErr('คืนค่าเริ่มต้นจากเซิร์ฟเวอร์แล้ว');
};

$('showBoxes').onchange = refreshOverlay;
$('showMarks').onchange = refreshOverlay;
$('showLandmarks').onchange = refreshOverlay;
$('showOrigin').onchange = refreshOverlay;
$('showGrid').onchange = refreshOverlay;
$('zoom').oninput = () => { $('overlay').style.width = $('zoom').value + 'px'; };
$('maskIndex').onchange = refreshMask;
$('onlyFlagged').onchange = () => lastData && renderTable(lastData);

function openTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.tabpane').forEach(p => { p.hidden = p.id !== 'tab-' + name; });
}
document.querySelectorAll('.tab').forEach(t => { t.onclick = () => openTab(t.dataset.tab); });

async function loadDefaults() {
  // select ไม่มีวันว่าง มันเลือก option แรกเสมอ ถ้าเช็คแค่ค่าว่างจะได้ค่าผิด
  // แล้วส่งทับค่าเริ่มต้นของเซิร์ฟเวอร์ทุกครั้ง จึงต้องเซ็ตค่าให้ select โดยตรง
  const cfg = await api('/api/config');
  for (const k of OPTS) {
    const el = $(k);
    const v = cfg[k];
    if (v === null || v === undefined) continue;
    if (el.tagName === 'SELECT') {
      if ([...el.options].some(o => o.value === String(v))) el.value = String(v);
    } else if (el.value === '') {
      el.value = v;
    }
  }
}

loadHealth();
loadDefaults();
loadPatients();
