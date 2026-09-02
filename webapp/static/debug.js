const BASE = location.pathname.replace(/\/[^/]*$/, '');
const u = (p) => BASE + p;
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const state = {};                       // ผลล่าสุดของแต่ละชุด ใช้เทียบสองฝั่ง

async function api(url, opts) {
  const res = await fetch(u(url), opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) { /* ไม่ใช่ json */ }
    throw new Error(msg);
  }
  return res.json();
}

// ค่าที่เปลี่ยนหน้าตาของภาพก่อนตีกรอบ — ดูผลได้ทันทีโดยไม่ต้องรันไปป์ไลน์
const PRE_KEYS = ['crop_pre', 'blackhat_thr', 'crop_pre_ksize', 'crop_pre_thr',
                  'crop_pre_hyst', 'crop_pre_close', 'crop_pre_dilate'];

const setErr = (m) => {
  const el = document.getElementById('err');
  el.textContent = m || '';
  el.hidden = !m;                       // แถบเตือนว่าง ๆ ไม่ควรกินที่
};
const chip = (label, value, cls = '') =>
  `<div class="chip ${cls}"><span>${label}</span> <b>${value}</b></div>`;

function overrides(sec) {
  const o = {};
  for (const el of $$('[data-k]', sec)) {
    const v = el.value;
    if (v !== '' && v !== null) o[el.dataset.k] = v;
  }
  return o;
}

async function loadDefaults() {
  const cfg = await api('/api/config');
  for (const el of $$('[data-k]')) {
    const v = cfg[el.dataset.k];
    if (v === null || v === undefined) continue;
    if (el.tagName === 'SELECT') {
      if ([...el.options].some(o => o.value === String(v))) el.value = String(v);
    } else if (el.value === '') {
      el.value = v;
    }
  }
}

async function refreshFiles(side) {
  const sec = document.querySelector(`.side[data-side="${side}"]`);
  const d = await api(`/api/debug/${side}/images`);
  const up = d.images.length, pk = (d.picked || []).length;
  sec.querySelector('.files').textContent = (up + pk)
    ? `รวม ${up + pk} ภาพ` + (up ? ` — อัปโหลด ${up}` : '') + (pk ? ` — จาก data/ ${pk}` : '')
    : 'ยังไม่มีภาพ';
  const sel = sec.querySelector('.dataList');
  const picked = new Set(d.picked || []);
  for (const o of sel.options) o.selected = picked.has(o.value);
  fillPrebinPick(sec, (d.items || []).map(i => i.name));
  return d;
}

function fillPrebinPick(sec, names) {
  const sel = sec.querySelector('.prebinPick');
  const keep = sel.value;
  sel.innerHTML = names.map(n => `<option value="${n}">${n}</option>`).join('');
  if (names.includes(keep)) sel.value = keep;
  refreshPrebin(sec);
}

function refreshPrebin(sec) {
  const side = sec.dataset.side;
  const name = sec.querySelector('.prebinPick').value;
  const img = sec.querySelector('.prebinImg');
  if (!name) { img.removeAttribute('src'); return; }
  const q = new URLSearchParams({ image: name, width: sec.querySelector('.prebinW').value });
  for (const k of PRE_KEYS) {
    const el = sec.querySelector(`[data-k="${k}"]`);
    if (el && el.value !== '' && el.value !== null) q.set(k, el.value);
  }
  img.src = u(`/api/debug/${side}/prebin?`) + q;
}

// รายชื่อภาพทั้งหมดที่มีในระบบ ใช้ร่วมกันทั้งสองฝั่ง
let catalog = [];

async function loadCatalog() {
  const [imgs, pats] = await Promise.all([api('/api/images'), api('/api/patients')]);
  const meta = {};
  for (const p of pats) for (const im of p.images) meta[im] = p;
  catalog = imgs.map(i => {
    const p = meta[i.name] || {};
    return { name: i.name, label: `${i.name}${p.name ? '  ·  ' + p.name : ''}${p.group ? '  ·  ' + p.group : ''}` };
  });
  for (const sec of $$('.side')) fillDataList(sec, '');
}

function fillDataList(sec, q) {
  const sel = sec.querySelector('.dataList');
  const keep = new Set([...sel.selectedOptions].map(o => o.value));
  const needle = q.trim().toLowerCase();
  const shown = catalog.filter(c => !needle || c.label.toLowerCase().includes(needle));
  sel.innerHTML = '';
  for (const c of shown) {
    const o = document.createElement('option');
    o.value = c.name; o.textContent = c.label; o.selected = keep.has(c.name);
    sel.appendChild(o);
  }
  sec.querySelector('.pickNote').textContent =
    `แสดง ${shown.length} จาก ${catalog.length} ภาพ`;
}

const N = (v, d = 0) => (v === null || v === undefined) ? '—' : Number(v).toFixed(d);

function renderSide(side, d) {
  state[side] = d;
  const sec = document.querySelector(`.side[data-side="${side}"]`);
  const a = d.aggregate;
  sec.querySelector('.sideAgg').innerHTML = [
    chip('ภาพ', a.n_images),
    chip('จังหวะรวม', a.beats),
    chip('จุด R รวม', a.r_peaks),
    chip('จากโมเดล', a.model, a.model ? 'good' : 'bad'),
    chip('จาก anchor', a.anchor),
    chip('HR เฉลี่ย', a.hr_mean ? `${N(a.hr_mean)} bpm` : '—'),
    chip('RR เฉลี่ย', a.rr_mean_mm ? `${N(a.rr_mean_mm, 2)} mm` : '—'),
    chip('flag รวม', a.flags, a.flags ? 'bad' : 'good'),
  ].join('');

  const head = ['ภาพ', 'จังหวะ', 'R', 'โมเดล', 'anchor', 'ซ้ำ', 'flag', 'px/mm', 'RR mm', 'HR'];
  sec.querySelector('.tbl').innerHTML =
    '<tr>' + head.map(h => `<th>${h}</th>`).join('') + '</tr>' +
    d.images.map(i => `<tr><td>${i.image}</td><td>${i.beats}</td><td>${i.r_peaks}</td>` +
      `<td>${i.model}</td><td>${i.anchor}</td><td>${i.dup}</td>` +
      `<td class="${i.flags ? 'flagged' : ''}">${i.flags}</td>` +
      `<td>${N(i.px_per_mm, 2)}</td><td>${N(i.rr_mean_mm, 2)}</td><td>${N(i.hr)}</td></tr>`).join('') +
    d.errors.map(e => `<tr class="flagged"><td>${e.image}</td><td colspan="9">${e.error}</td></tr>`).join('');

  sec.querySelector('.shots').innerHTML = d.images.map(i => {
    const q = `image=${encodeURIComponent(i.image)}&rev=${i.rev}`;
    return `<figure class="shot">
      <figcaption>${i.image}</figcaption>
      <img alt="" src="${u(`/api/debug/${side}/overlay`)}?${q}&width=900">
      <figcaption class="hint">ครอปที่ป้อนโมเดลจุด (+ = ตำแหน่งที่ชุดเทรนวาง R)</figcaption>
      <img alt="" src="${u(`/api/debug/${side}/crops`)}?${q}&n=5&size=200">
    </figure>`;
  }).join('');

  renderCompare();
}

function renderCompare() {
  const a = state.train && state.train.aggregate;
  const b = state.test && state.test.aggregate;
  if (!a || !b) {
    document.getElementById('agg').innerHTML =
      '<p class="hint">รันให้ครบทั้งสองชุดเพื่อดูผลเทียบ</p>';
    return;
  }
  const ratio = (x, y) => (y ? x / y : null);
  const modelShare = (g) => g.r_peaks ? 100 * g.model / g.r_peaks : null;
  const flagRate = (g) => g.r_peaks ? 100 * g.flags / g.r_peaks : null;
  const gap = (f) => {
    const x = f(a), y = f(b);
    if (x === null || y === null) return '—';
    return `${N(x, 1)}% / ${N(y, 1)}%`;
  };
  document.getElementById('agg').innerHTML = [
    chip('จุด R เทรน / เทส', `${a.r_peaks} / ${b.r_peaks}`),
    chip('โมเดลยืนยัน (สัดส่วน)', gap(modelShare),
         Math.abs(modelShare(a) - modelShare(b)) > 15 ? 'bad' : 'good'),
    chip('flag ต่อจุด', gap(flagRate),
         Math.abs(flagRate(a) - flagRate(b)) > 5 ? 'bad' : 'good'),
    chip('HR เฉลี่ย', `${N(a.hr_mean)} / ${N(b.hr_mean)}`),
    chip('RR เฉลี่ย mm', `${N(a.rr_mean_mm, 2)} / ${N(b.rr_mean_mm, 2)}`),
  ].join('') +
    '<p class="hint">ตัวเลขคู่คือ เทรน / เทส — ช่องแดงคือจุดที่สองชุดต่างกันมากพอจะเป็นสัญญาณว่าหลุดโดเมน</p>';
}

for (const sec of $$('.side')) {
  const side = sec.dataset.side;

  sec.querySelector('.pick').onchange = async (e) => {
    const files = [...e.target.files];
    e.target.value = '';
    if (!files.length) return;
    setErr('');
    const fd = new FormData();
    files.forEach(f => fd.append('files', f));
    try {
      const r = await api(`/api/debug/${side}/images`, { method: 'POST', body: fd });
      await refreshFiles(side);
      if (r.failed.length) setErr(r.failed.map(f => `${f.file}: ${f.reason}`).join('\n'));
    } catch (err) { setErr(err.message); }
  };

  sec.querySelector('.clear').onclick = async () => {
    if (!confirm(`ล้างภาพทั้งหมดของชุด ${side}?`)) return;
    try {
      await api(`/api/debug/${side}/images`, { method: 'DELETE' });
      delete state[side];
      await refreshFiles(side);
      sec.querySelector('.sideAgg').innerHTML = '';
      sec.querySelector('.tbl').innerHTML = '';
      sec.querySelector('.shots').innerHTML = '';
      renderCompare();
    } catch (err) { setErr(err.message); }
  };

  sec.querySelector('.prebinPick').onchange = () => refreshPrebin(sec);
  sec.querySelector('.prebinW').oninput = () => refreshPrebin(sec);
  for (const k of PRE_KEYS) {
    const el = sec.querySelector(`[data-k="${k}"]`);
    if (el) el.onchange = () => refreshPrebin(sec);
  }

  sec.querySelector('.filter').oninput = (e) => fillDataList(sec, e.target.value);
  sec.querySelector('.selAll').onclick = () => {
    for (const o of sec.querySelector('.dataList').options) o.selected = true;
  };
  sec.querySelector('.selNone').onclick = () => {
    for (const o of sec.querySelector('.dataList').options) o.selected = false;
  };
  sec.querySelector('.usePicked').onclick = async () => {
    const names = [...sec.querySelector('.dataList').selectedOptions].map(o => o.value);
    setErr('');
    try {
      await api(`/api/debug/${side}/pick`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ images: names }),
      });
      await refreshFiles(side);
    } catch (err) { setErr(err.message); }
  };

  sec.querySelector('.run').onclick = async () => {
    const btn = sec.querySelector('.run');
    btn.disabled = true; btn.textContent = 'กำลังรัน...';
    setErr('');
    try {
      renderSide(side, await api(`/api/debug/${side}/run`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ overrides: overrides(sec) }),
      }));
    } catch (err) { setErr(err.message); }
    finally { btn.disabled = false; btn.textContent = 'รันชุดนี้'; }
  };

  refreshFiles(side);
}
loadCatalog();
loadDefaults();
renderCompare();
