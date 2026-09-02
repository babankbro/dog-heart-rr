const BASE = location.pathname.replace(/\/[^/]*$/, '');
const u = (p) => BASE + p;
const $ = (id) => document.getElementById(id);
let data = null;
let sortKey = 'id';
let sortDir = 1;

const N = (v, d = 2) => (v === null || v === undefined) ? '—' : Number(v).toFixed(d);
const chip = (label, value, cls = '') =>
  `<div class="chip ${cls}"><span>${label}</span> <b>${value}</b></div>`;
const val = (p, m) => (p.summary ? p.summary[m].value : -1);

function method(p, m) {
  if (!p.summary) return '—';
  const s = p.summary[m];
  return `${N(s.value)} <span class="sem">± ${N(s.sem)}</span>` +
         (m === 'mid20' ? ` <span class="sem">n=${s.n_used}</span>` : '');
}

const COLS = [
  { k: 'id', t: 'รหัส', get: p => p.id },
  { k: 'name', t: 'ชื่อ', get: p => p.name || '—' },
  { k: 'group', t: 'ประเภท', get: p => p.group || '—' },
  { k: 'images', t: 'ภาพที่ใช้', get: p => `${p.images_used.length}/${p.n_images}`,
    sort: p => p.images_used.length },
  { k: 'n', t: 'ช่วง RR', get: p => p.summary ? p.summary.n : 0,
    sort: p => p.summary ? p.summary.n : -1 },
  { k: 'min', t: 'ต่ำสุด', get: p => p.summary ? N(p.summary.min) : '—',
    sort: p => p.summary ? p.summary.min : -1 },
  { k: 'max', t: 'สูงสุด', get: p => p.summary ? N(p.summary.max) : '—',
    sort: p => p.summary ? p.summary.max : -1 },
  { k: 'sd', t: 'SD', get: p => p.summary ? N(p.summary.sd) : '—',
    sort: p => p.summary ? p.summary.sd : -1 },
  { k: 'mean_all', t: 'mean_all', get: p => method(p, 'mean_all'), sort: p => val(p, 'mean_all') },
  { k: 'mid20', t: 'mid20', get: p => method(p, 'mid20'), sort: p => val(p, 'mid20') },
  { k: 'median', t: 'median', get: p => method(p, 'median'), sort: p => val(p, 'median') },
  { k: 'spread', t: 'ต่างสุด',
    get: p => p.summary ? `${N(p.summary.spread)} (${N(p.summary.spread_pct, 1)}%)` : '—',
    sort: p => p.summary ? p.summary.spread : -1 },
];

function render() {
  const ps = [...data.patients];
  const col = COLS.find(c => c.k === sortKey) || COLS[0];
  const key = col.sort || col.get;
  ps.sort((a, b) => {
    const x = key(a), y = key(b);
    return (typeof x === 'string' ? x.localeCompare(y) : x - y) * sortDir;
  });

  const withData = ps.filter(p => p.summary);
  const spreads = withData.map(p => p.summary.spread_pct).sort((a, b) => a - b);
  const dropped = ps.reduce((n, p) => n + p.images_dropped.length, 0);
  const pending = ps.reduce((n, p) => n + p.images_pending.length, 0);
  $('agg').innerHTML = [
    chip('สัตว์ทั้งหมด', ps.length),
    chip('มีข้อมูล RR', withData.length, withData.length === ps.length ? 'good' : 'bad'),
    chip('ช่วง RR รวม', withData.reduce((n, p) => n + p.summary.n, 0)),
    chip('ภาพที่ถูกตัดออก', dropped, dropped ? 'bad' : 'good'),
    chip('ภาพที่ยังไม่วิเคราะห์', pending, pending ? 'bad' : 'good'),
    chip('ต่างสุดระหว่างวิธี (มัธยฐาน)',
         spreads.length ? `${N(spreads[Math.floor(spreads.length / 2)], 1)}%` : '—'),
  ].join('');

  const noData = ps.filter(p => !p.summary);
  $('warn').hidden = !noData.length;
  $('warn').innerHTML = noData.length
    ? `⚠ ${noData.length} ตัวไม่มีค่า RR ที่ใช้ได้เลย เพราะทุกภาพของตัวนั้นวัดสเกลผิด: ` +
      noData.map(p => `${p.id} ${p.name}`).join(', ')
    : '';

  $('tbl').innerHTML =
    '<tr>' + COLS.map(c =>
      `<th class="sortable${c.k === sortKey ? ' on' : ''}" data-k="${c.k}">${c.t}` +
      (c.k === sortKey ? (sortDir > 0 ? ' ▲' : ' ▼') : '') + '</th>').join('') + '</tr>' +
    ps.map(p => {
      const wide = p.summary && p.summary.spread_pct > 10;
      const title = p.images_dropped.length
        ? ` title="ตัดออก: ${p.images_dropped.map(d => d.image + ' — ' + d.reason).join(' | ')}"`
        : '';
      return `<tr class="${!p.summary ? 'flagged' : (wide ? 'anchor' : '')}"${title}>` +
        COLS.map(c => `<td>${c.get(p)}</td>`).join('') + '</tr>';
    }).join('');

  for (const th of document.querySelectorAll('#tbl th.sortable')) {
    th.onclick = () => {
      const k = th.dataset.k;
      sortDir = (k === sortKey) ? -sortDir : 1;
      sortKey = k;
      render();
    };
  }
}

async function load() {
  const q = $('excl').checked ? 1 : 0;
  $('csv').href = u(`/api/rr-summary.csv?exclude_bad_scale=${q}`);
  const res = await fetch(u(`/api/rr-summary?exclude_bad_scale=${q}`));
  data = await res.json();
  $('note').textContent = $('excl').checked
    ? 'ภาพที่ HR อยู่นอกช่วง 40–300 bpm ถูกตัดออก เพราะ px/mm ของภาพนั้นน่าจะวัดผิด'
    : 'รวมทุกภาพ รวมภาพที่สเกลน่าสงสัยด้วย ตัวเลขอาจเพี้ยน';
  render();
}

$('excl').onchange = load;
load();
