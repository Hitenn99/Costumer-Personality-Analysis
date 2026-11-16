/* ==========================================================================
   Customer Personality Analysis Dashboard
   All numbers are computed in the browser from data/customers.json and
   data/model_metrics.json (written by ml/run_pipeline.py).
   ========================================================================== */
'use strict';

/* ---------- Design tokens ---------- */
const C = {
  orange: '#F28C28', orange2: '#F7B26A', orange3: '#FBD7AE', orange4: '#FDEBD3', orangePale: '#FDE3C4',
  slate: '#3B4246', grey: '#9AA0A6', greyLight: '#E3E5E7', text: '#2E2E2E', muted: '#8A8F94',
  up: '#2E9E5B', down: '#E74C3C', faint: '#E9EAEB',
};
const SEGMENT_PALETTE = ['#F28C28', '#3B4246', '#F7B26A', '#9AA0A6', '#B5651D', '#6F7C82'];
const FONT = 'Inter, system-ui, sans-serif';

const TOOLTIP = {
  backgroundColor: '#FFFFFF',
  borderColor: '#E6E6E6',
  borderWidth: 1,
  padding: [8, 10],
  textStyle: { color: C.text, fontFamily: FONT, fontSize: 12 },
  extraCssText: 'box-shadow:0 2px 8px rgba(0,0,0,.12);border-radius:6px;',
};

/* Preferred display orders for categorical values (only values present in the data are used). */
const ORDERS = {
  Education: ['Undergraduate', 'Graduate', 'Postgraduate', 'PhD'],
  Marital_Status: ['Couple', 'Single'],
  Age_Group: ['<30', '30-44', '45-59', '60-74', '75+'],
  Income_Band: ['Low (<30k)', 'Mid (30-60k)', 'High (60-90k)', 'Very High (90k+)'],
  Response: ['1', '0'],
  Complain: ['1', '0'],
};
const yesNo = v => (v === '1' ? 'Yes' : 'No');

const FILTER_GROUPS = [
  { key: 'cluster_name', label: 'Segment' },
  { key: 'Education', label: 'Education' },
  { key: 'Marital_Status', label: 'Marital Status' },
  { key: 'Age_Group', label: 'Age Group' },
  { key: 'Income_Band', label: 'Income Band', display: v => (v.match(/\((.*)\)/) || [, v])[1] },
  { key: 'Total_Children', label: 'Children' },
  { key: 'Response', label: 'Response', display: yesNo },
  { key: 'Complain', label: 'Complain', display: yesNo },
];

const PRODUCTS = [
  ['Wines', 'MntWines'], ['Meat', 'MntMeatProducts'], ['Gold', 'MntGoldProds'],
  ['Fish', 'MntFishProducts'], ['Sweets', 'MntSweetProducts'], ['Fruits', 'MntFruits'],
];
const CHANNEL_METRICS = [
  ['Web Purchases', 'NumWebPurchases'], ['Catalog Purchases', 'NumCatalogPurchases'],
  ['Store Purchases', 'NumStorePurchases'], ['Deals Purchases', 'NumDealsPurchases'],
  ['Web Visits', 'NumWebVisitsMonth'],
];
const PERIOD_NAMES = { W: 'week', M: 'month', Q: 'quarter', Y: 'year' };

/* ---------- Global state ---------- */
const state = {
  customers: [],
  metrics: null,
  filters: {},        // key -> Set of string values
  options: {},        // key -> ordered list of string values
  segments: [],       // [{id, name}]
  segColor: {},       // name -> colour
  quartiles: {},      // channel metric field -> [t1, t2, t3]
  trendPeriod: 'M',
  kpi: {},
  charts: {},
};

/* ---------- Helpers ---------- */
const $ = id => document.getElementById(id);
const fmtInt = n => Math.round(n).toLocaleString('en-US');
const fmtPct = (n, d = 1) => `${(n * 100).toFixed(d)}%`;
function fmtMoney(n) {
  const a = Math.abs(n);
  if (a >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (a >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${Math.round(n)}`;
}
const toDate = iso => new Date(`${iso}T00:00:00Z`);
const fmtDate = iso => toDate(iso).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric', timeZone: 'UTC' });
const escapeHtml = s => String(s).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
const sum = (arr, f) => arr.reduce((acc, d) => acc + f(d), 0);
const mean = (arr, f) => (arr.length ? sum(arr, f) / arr.length : 0);
const custId = id => `C_${id}`;

function lerpColor(a, b, t) {
  const pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16);
  const ch = (p, s) => (p >> s) & 255;
  const mix = s => Math.round(ch(pa, s) + (ch(pb, s) - ch(pa, s)) * t);
  return `rgb(${mix(16)},${mix(8)},${mix(0)})`;
}

function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function getChart(id) {
  if (!state.charts[id]) {
    const el = $(id);
    const chart = echarts.init(el, null, { renderer: 'canvas' });
    // Cards are sized by the viewport-fitted grid, so redraw whenever the box changes.
    let raf = 0;
    new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => chart.resize());
    }).observe(el);
    state.charts[id] = chart;
  }
  return state.charts[id];
}
function onceClick(chart, handler) {
  if (chart.__clickBound) return;
  chart.on('click', handler);
  chart.__clickBound = true;
}

function animateNumber(el, key, to, fmt) {
  const from = state.kpi[key] ?? 0;
  state.kpi[key] = to;
  cancelAnimationFrame(el._raf);
  const t0 = performance.now(), dur = 650;
  const step = now => {
    const p = Math.min(1, (now - t0) / dur);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = fmt(from + (to - from) * eased);
    if (p < 1) el._raf = requestAnimationFrame(step);
  };
  el._raf = requestAnimationFrame(step);
}

/* ---------- Filtering ---------- */
function filterData(exceptKey = null) {
  const active = FILTER_GROUPS.filter(g => g.key !== exceptKey && state.filters[g.key].size > 0);
  if (!active.length) return state.customers;
  return state.customers.filter(c => active.every(g => state.filters[g.key].has(String(c[g.key]))));
}

function toggleFilter(key, value) {
  const set = state.filters[key];
  const v = String(value);
  if (set.has(v)) set.delete(v); else set.add(v);
  syncChips();
  renderAll();
}

function resetFilters() {
  FILTER_GROUPS.forEach(g => state.filters[g.key].clear());
  syncChips();
  renderAll();
}

/* ---------- Initialisation ---------- */
const fetchJson = name => fetch(`data/${name}`, { cache: 'no-store' })
  .then(r => { if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`); return r.json(); });

async function init() {
  window.addEventListener('hashchange', route);
  try {
    const [customers, metrics] = await Promise.all([fetchJson('customers.json'), fetchJson('model_metrics.json')]);
    state.customers = customers;
    state.metrics = metrics;
  } catch (err) {
    const box = $('load-error');
    box.hidden = false;
    box.textContent = `Could not load dashboard data (${err.message}).\n` +
      'Run "python ml/run_pipeline.py" from the project root, then start the app with ' +
      '"python server.py" and open http://localhost:8000. You can also upload a dataset on the Dataset page.';
    $('grid').hidden = true;
    if (typeof initDataset === 'function') initDataset();
    route();
    return;
  }

  buildOptions();
  buildChips();
  buildQuartiles();
  buildRecentDateOptions();
  bindControls();

  const dr = state.metrics.date_range;
  $('date-range').textContent = `${fmtDate(dr.min)} to ${fmtDate(dr.max)}`;
  const ds = state.metrics.dataset;
  if (ds) $('nav-dataset').innerHTML = `Dataset: <strong>${escapeHtml(ds.name)}</strong> · ${fmtInt(state.customers.length)} customers`;

  renderStatic();
  renderAll();
  if (typeof initDataset === 'function') initDataset();
  if (typeof initAI === 'function') initAI();
  route();
  showPendingToast();
}

/* ---------- Page routing (#/dashboard, #/dataset, #/ai?id=123) ---------- */
const PAGES = ['dashboard', 'dataset', 'ai'];

function parseHash() {
  const [path, query = ''] = location.hash.replace(/^#\/?/, '').split('?');
  return { page: PAGES.includes(path) ? path : 'dashboard', params: new URLSearchParams(query) };
}

function route() {
  const { page, params } = parseHash();
  PAGES.forEach(p => { $(`page-${p}`).hidden = p !== page; });
  document.querySelectorAll('.topnav .tab').forEach(t => t.classList.toggle('active', t.dataset.page === page));
  document.body.dataset.page = page;
  if (page === 'dataset' && typeof showDatasetPage === 'function') showDatasetPage();
  if (page === 'ai' && typeof showAIPage === 'function') showAIPage(params.get('id'));
}

function navigate(page, params = '') {
  location.hash = `#/${page}${params ? `?${params}` : ''}`;
}

/* ---------- Toast (also survives a reload via sessionStorage) ---------- */
function toast(message, kind = 'ok', ms = 5000) {
  const el = $('toast');
  el.className = `toast toast--${kind}`;
  el.textContent = message;
  el.hidden = false;
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.hidden = true; }, ms);
}
function showPendingToast() {
  try {
    const msg = sessionStorage.getItem('pendingToast');
    if (msg) { sessionStorage.removeItem('pendingToast'); toast(msg); }
  } catch { /* storage unavailable */ }
}

function buildOptions() {
  const segMap = new Map();
  state.customers.forEach(c => segMap.set(c.cluster_id, c.cluster_name));
  state.segments = [...segMap.entries()].sort((a, b) => a[0] - b[0]).map(([id, name]) => ({ id, name }));
  state.segments.forEach((s, i) => { state.segColor[s.name] = SEGMENT_PALETTE[i % SEGMENT_PALETTE.length]; });

  FILTER_GROUPS.forEach(g => {
    state.filters[g.key] = new Set();
    const present = new Set(state.customers.map(c => String(c[g.key])));
    let opts;
    if (g.key === 'cluster_name') opts = state.segments.map(s => s.name);
    else if (ORDERS[g.key]) opts = ORDERS[g.key].filter(v => present.has(v));
    else opts = [...present].sort((a, b) => Number(a) - Number(b));
    state.options[g.key] = opts;
  });
}

function buildChips() {
  const host = $('filter-groups');
  host.innerHTML = FILTER_GROUPS.map(g => `
    <div class="fgroup">
      <div class="fgroup__label">${escapeHtml(g.label)}</div>
      <div class="fgroup__chips">
        ${state.options[g.key].map(v => `<button class="chip" data-key="${g.key}" data-value="${escapeHtml(v)}">${escapeHtml(g.display ? g.display(v) : v)}</button>`).join('')}
      </div>
    </div>`).join('');
  host.addEventListener('click', e => {
    const chip = e.target.closest('.chip');
    if (chip) toggleFilter(chip.dataset.key, chip.dataset.value);
  });
}

function syncChips() {
  document.querySelectorAll('.chip').forEach(chip => {
    chip.classList.toggle('active', state.filters[chip.dataset.key].has(chip.dataset.value));
  });
}

function buildQuartiles() {
  CHANNEL_METRICS.forEach(([, field]) => {
    const sorted = state.customers.map(c => c[field]).sort((a, b) => a - b);
    state.quartiles[field] = [0.25, 0.5, 0.75].map(q => quantile(sorted, q));
  });
}

function buildRecentDateOptions() {
  const years = [...new Set(state.customers.map(c => c.Dt_Customer.slice(0, 4)))].sort().reverse();
  const sel = $('recent-date');
  sel.innerHTML = '<option value="all">All</option><option value="30">Last 30 days</option><option value="90">Last 90 days</option>' +
    years.map(y => `<option value="y${y}">${y}</option>`).join('');
}

function bindControls() {
  $('reset-filters').addEventListener('click', e => { e.preventDefault(); resetFilters(); });
  $('empty-reset').addEventListener('click', e => { e.preventDefault(); resetFilters(); });
  $('recent-id').addEventListener('input', () => renderRecent(filterData()));
  $('recent-date').addEventListener('change', () => renderRecent(filterData()));
  $('trend-buttons').addEventListener('click', e => {
    const btn = e.target.closest('button');
    if (!btn) return;
    state.trendPeriod = btn.dataset.period;
    document.querySelectorAll('#trend-buttons button').forEach(b => b.classList.toggle('active', b === btn));
    renderTrend(filterData());
  });
}

/* ---------- Render orchestration ---------- */
function renderAll() {
  const data = filterData();
  $('empty-banner').hidden = data.length > 0;

  renderKPIs(data);
  renderMarital();
  renderAgeGroups();
  renderEducation();
  renderSegments();
  renderProducts(data);
  renderChannel(data);
  renderRecent(data);
  renderTrend(data);
  renderLatent(data);
  renderProfile(data);
  renderAnomaly(data);
}

/** Model charts that do not depend on filters. */
function renderStatic() {
  renderLoss();
  renderOptimalK();
  renderComparison();
}

/* ---------- KPIs ---------- */
function renderKPIs(data) {
  const responders = sum(data, d => d.Response);
  const rate = data.length ? responders / data.length : 0;
  animateNumber($('kpi-rate'), 'rate', rate * 100, v => `${v.toFixed(1)}%`);
  animateNumber($('kpi-responders'), 'responders', responders, fmtInt);
  animateNumber($('kpi-customers'), 'customers', data.length, fmtInt);
}

/* ---------- Donuts (marital + segment) ---------- */
function donutOption(total, resp) {
  const empty = total === 0;
  return {
    textStyle: { fontFamily: FONT },
    tooltip: {
      ...TOOLTIP, trigger: 'item', confine: true,
      formatter: p => (empty ? 'No customers' : `${p.marker}${p.name}: <b>${fmtInt(p.value)}</b> (${fmtPct(p.value / total)})`),
    },
    series: [{
      type: 'pie',
      radius: ['56%', '80%'],
      startAngle: 90,
      label: { show: false },
      labelLine: { show: false },
      emphasis: { scale: false },
      animationDuration: 500,
      data: empty
        ? [{ value: 1, name: 'None', itemStyle: { color: C.faint } }]
        : [
          { value: resp, name: 'Responders', itemStyle: { color: C.orange } },
          { value: total - resp, name: 'Non-responders', itemStyle: { color: C.slate } },
        ],
    }],
    graphic: [{
      type: 'text', left: 'center', top: 'middle',
      style: { text: fmtInt(resp), fill: C.orange, font: `700 13px ${FONT}` },
    }],
  };
}

function renderDonutList(hostId, key, items, data) {
  const host = $(hostId);
  const sel = state.filters[key];
  items.forEach((item, i) => {
    const rowId = `${hostId}-row-${i}`;
    let row = $(rowId);
    if (!row) {
      row = document.createElement('div');
      row.className = 'donut-row';
      row.id = rowId;
      row.title = `Click to filter by ${item.label}`;
      row.innerHTML = `
        <div class="donut-row__label">
          <div class="donut-row__name">${escapeHtml(item.label)}</div>
          <div class="donut-row__total"></div>
        </div>
        <div class="donut-row__chart" id="${rowId}-chart"></div>
        <span class="donut-row__resp"></span>`;
      row.addEventListener('click', () => toggleFilter(key, item.value));
      host.appendChild(row);
    }
    const subset = data.filter(d => String(d[key]) === item.value);
    const total = subset.length;
    const resp = sum(subset, d => d.Response);
    row.querySelector('.donut-row__total').textContent = fmtInt(total);
    row.querySelector('.donut-row__resp').textContent = fmtInt(resp);
    row.classList.toggle('selected', sel.has(item.value));
    row.classList.toggle('dimmed', sel.size > 0 && !sel.has(item.value));
    getChart(`${rowId}-chart`).setOption(donutOption(total, resp), true);
  });
}

function renderMarital() {
  const items = state.options.Marital_Status.map(v => ({ value: v, label: v }));
  renderDonutList('marital-donuts', 'Marital_Status', items, filterData('Marital_Status'));
}

function renderSegments() {
  const items = state.segments.map(s => ({ value: s.name, label: s.name }));
  renderDonutList('segment-donuts', 'cluster_name', items, filterData('cluster_name'));
}

/* ---------- Paired bars (age / education) ---------- */
function pairedBarOption(categories, totals, responders, selected, horizontal) {
  const dim = v => selected.size > 0 && !selected.has(v);
  const mkData = (vals, color) => vals.map((v, i) => ({
    value: v,
    itemStyle: { color, opacity: dim(categories[i]) ? 0.3 : 1, borderRadius: horizontal ? [0, 3, 3, 0] : [3, 3, 0, 0] },
  }));
  const label = { show: true, position: horizontal ? 'right' : 'top', color: C.muted, fontSize: 10, formatter: p => fmtInt(p.value) };
  const catAxis = {
    type: 'category', data: categories, inverse: horizontal,
    axisLine: { show: !horizontal, lineStyle: { color: C.greyLight } },
    axisTick: { show: false },
    axisLabel: { color: C.muted, fontSize: 11 },
  };
  const valAxis = { type: 'value', show: false, splitLine: { show: false } };
  return {
    textStyle: { fontFamily: FONT },
    grid: horizontal ? { left: 4, right: 38, top: 4, bottom: 4, containLabel: true } : { left: 4, right: 4, top: 18, bottom: 4, containLabel: true },
    tooltip: {
      ...TOOLTIP, trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(0,0,0,.03)' } },
      formatter: ps => {
        const tot = ps.find(p => p.seriesName === 'Customers')?.value ?? 0;
        const res = ps.find(p => p.seriesName === 'Responders')?.value ?? 0;
        return `<b>${escapeHtml(ps[0].name)}</b><br/>Customers: <b>${fmtInt(tot)}</b><br/>Responders: <b>${fmtInt(res)}</b>` +
          (tot ? ` (${fmtPct(res / tot)})` : '');
      },
    },
    xAxis: horizontal ? valAxis : catAxis,
    yAxis: horizontal ? catAxis : valAxis,
    series: [
      { name: 'Customers', type: 'bar', barGap: '12%', barMaxWidth: 22, data: mkData(totals, C.slate), label },
      { name: 'Responders', type: 'bar', barMaxWidth: 22, data: mkData(responders, C.orange), label },
    ],
  };
}

function renderPairedBars(chartId, key, horizontal) {
  const data = filterData(key);
  const cats = state.options[key];
  const totals = cats.map(c => data.filter(d => d[key] === c).length);
  const resp = cats.map(c => data.filter(d => d[key] === c && d.Response === 1).length);
  const chart = getChart(chartId);
  chart.setOption(pairedBarOption(cats, totals, resp, state.filters[key], horizontal), true);
  bindCategoryClick(chart, key, horizontal);
}

/** Clicking anywhere in a category's column (bar, label or gap) toggles that category. */
function bindCategoryClick(chart, key, horizontal) {
  if (chart.__clickBound) return;
  chart.__clickBound = true;
  chart.getZr().on('click', e => {
    const pt = [e.offsetX, e.offsetY];
    if (!chart.containPixel('grid', pt)) return;
    const v = chart.convertFromPixel('grid', pt);
    const idx = Math.round(horizontal ? v[1] : v[0]);
    const cats = state.options[key];
    if (idx >= 0 && idx < cats.length) toggleFilter(key, cats[idx]);
  });
}
const renderAgeGroups = () => renderPairedBars('chart-age', 'Age_Group', false);
const renderEducation = () => renderPairedBars('chart-education', 'Education', true);

/* ---------- Product category ---------- */
function renderProducts(data) {
  const rows = PRODUCTS.map(([name, field]) => ({
    name,
    total: sum(data, d => d[field]),
    resp: sum(data, d => (d.Response === 1 ? d[field] : 0)),
  })).sort((a, b) => b.total - a.total);
  const grand = sum(rows, r => r.total);
  const maxShare = Math.max(...rows.map(r => (grand ? r.total / grand : 0)), 1e-9);

  $('products').innerHTML = rows.map(r => {
    const share = grand ? r.total / grand : 0;
    const width = (share / maxShare) * 100;
    const respWidth = r.total ? width * (r.resp / r.total) : 0;
    return `
      <div class="prod" title="${escapeHtml(r.name)}: ${fmtMoney(r.total)} total, ${fmtMoney(r.resp)} from responders (${r.total ? fmtPct(r.resp / r.total) : '0%'})">
        <span class="prod__name">${escapeHtml(r.name)}</span>
        <span class="badge badge--orange">${fmtMoney(r.resp)}</span>
        <span class="badge badge--slate">${fmtMoney(r.total)}</span>
        <div class="prod__track">
          <div class="prod__fill" style="width:${width}%"></div>
          <div class="prod__fill-resp" style="width:${respWidth}%"></div>
        </div>
        <span class="prod__share">${fmtPct(share)}</span>
      </div>`;
  }).join('');
}

/* ---------- Channel behaviour heatmap ---------- */
function bucketOf(value, th) {
  if (value <= th[0]) return 0;
  if (value <= th[1]) return 1;
  if (value <= th[2]) return 2;
  return 3;
}
function bucketRange(field, q) {
  const th = state.quartiles[field];
  const r = v => (Number.isInteger(v) ? v : v.toFixed(1));
  if (q === 0) return `≤ ${r(th[0])}`;
  if (q === 3) return `> ${r(th[2])}`;
  return `> ${r(th[q - 1])} and ≤ ${r(th[q])}`;
}

function renderChannel(data) {
  // y-categories: for each metric an orange "R" row followed by a grey "N" row
  const yCats = [];
  const cells = [];
  const groups = [
    { suffix: 'R', test: d => d.Response === 1, from: '#FEF4E8', to: C.orange, label: 'Responders' },
    { suffix: 'N', test: d => d.Response !== 1, from: '#F1F2F3', to: C.slate, label: 'Non-responders' },
  ];
  CHANNEL_METRICS.forEach(([name, field]) => {
    groups.forEach(g => {
      const y = yCats.length;
      yCats.push(`${name}|${g.suffix}`);
      const counts = [0, 0, 0, 0];
      data.filter(g.test).forEach(d => { counts[bucketOf(d[field], state.quartiles[field])] += 1; });
      const rowMax = Math.max(...counts, 1);
      const rowTotal = sum(counts, v => v);
      counts.forEach((v, x) => {
        const t = v / rowMax;
        cells.push({
          value: [x, y, v],
          color: lerpColor(g.from, g.to, v === 0 ? 0 : 0.15 + 0.85 * t),
          textColor: t > 0.55 ? '#FFFFFF' : C.text,
          tip: `<b>${escapeHtml(name)}</b> · ${g.label}<br/>Q${x + 1} range: ${bucketRange(field, x)}<br/>` +
               `Customers: <b>${fmtInt(v)}</b>${rowTotal ? ` (${fmtPct(v / rowTotal)} of ${g.label.toLowerCase()})` : ''}`,
        });
      });
    });
  });

  getChart('chart-channel').setOption({
    textStyle: { fontFamily: FONT },
    grid: { left: 4, right: 6, top: 28, bottom: 6, containLabel: true },
    tooltip: { ...TOOLTIP, trigger: 'item', formatter: p => cells[p.dataIndex].tip },
    xAxis: {
      type: 'category', position: 'top', data: ['Q1 · Low', 'Q2', 'Q3', 'Q4 · High'],
      axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: C.muted, fontSize: 11 },
    },
    yAxis: {
      type: 'category', inverse: true, data: yCats,
      axisLine: { show: false }, axisTick: { show: false },
      axisLabel: {
        fontSize: 11, color: C.slate,
        formatter: v => (v.endsWith('|R') ? v.slice(0, -2) : ''),
      },
    },
    series: [{
      type: 'custom',
      encode: { x: 0, y: 1 },
      data: cells.map(c => c.value),
      renderItem: (params, api) => {
        const cell = cells[params.dataIndex];
        const center = api.coord([api.value(0), api.value(1)]);
        const size = api.size([1, 1]);
        // Pair each orange row with its grey row (2px apart); pairs sit further apart.
        const pairGap = Math.min(10, size[1] * 0.3);
        const w = size[0] - 4, h = size[1] - 2 - pairGap;
        const cy = center[1] + (api.value(1) % 2 === 0 ? pairGap / 2 : -pairGap / 2);
        const fontSize = Math.max(9, Math.min(12, h * 0.5));
        return {
          type: 'group',
          children: [
            { type: 'rect', shape: { x: center[0] - w / 2, y: cy - h / 2, width: w, height: h, r: 4 }, style: { fill: cell.color } },
            { type: 'text', style: { text: fmtInt(api.value(2)), x: center[0], y: cy, align: 'center', verticalAlign: 'middle', fill: cell.textColor, font: `600 ${fontSize}px ${FONT}` } },
          ],
        };
      },
    }],
  }, true);
}

/* ---------- Recent responders ---------- */
function renderRecent(data) {
  const idQuery = $('recent-id').value.trim().replace(/^c_?/i, '');
  const dateSel = $('recent-date').value;
  const maxDate = toDate(state.metrics.date_range.max);
  const dayMs = 86400000;
  const dateOk = c => {
    if (dateSel === 'all') return true;
    if (dateSel.startsWith('y')) return c.Dt_Customer.startsWith(dateSel.slice(1));
    return (maxDate - toDate(c.Dt_Customer)) / dayMs <= Number(dateSel);
  };
  const list = data
    .filter(c => c.Response === 1 && dateOk(c) && (!idQuery || String(c.ID).includes(idQuery)))
    .sort((a, b) => (a.Dt_Customer < b.Dt_Customer ? 1 : a.Dt_Customer > b.Dt_Customer ? -1 : b.ID - a.ID))
    .slice(0, 20);

  if (!list.length) {
    $('recent-list').innerHTML = '<div class="muted-empty">No responders match.</div>';
    return;
  }
  $('recent-list').innerHTML = list.map(c => `
    <div class="rcard">
      <div class="rcard__top">
        <span class="rcard__id">${custId(c.ID)}</span>
        <span class="rcard__date">${fmtDate(c.Dt_Customer)}</span>
      </div>
      <div class="rcard__seg"><span class="seg-dot" style="background:${state.segColor[c.cluster_name]}"></span>${escapeHtml(c.cluster_name)} · ${escapeHtml(c.Education)}</div>
      <div class="rcard__stats">
        <div><div class="stat__k">Income</div><div class="stat__v">${fmtMoney(c.Income)}</div></div>
        <div><div class="stat__k">Spending</div><div class="stat__v">${fmtMoney(c.Total_Spending)}</div></div>
        <div><div class="stat__k">Recency</div><div class="stat__v">${c.Recency} days</div></div>
        <div><div class="stat__k">Campaigns</div><div class="stat__v">${c.Total_Campaigns_Accepted} of 5</div></div>
        <div><div class="stat__k">Purchases</div><div class="stat__v">${c.Total_Purchases}</div></div>
        <div><div class="stat__k">Age</div><div class="stat__v">${c.Age}</div></div>
      </div>
    </div>`).join('');
}

/* ---------- Enrollment trend ---------- */
function periodStart(d, p) {
  const y = d.getUTCFullYear(), m = d.getUTCMonth();
  if (p === 'Y') return Date.UTC(y, 0, 1);
  if (p === 'Q') return Date.UTC(y, m - (m % 3), 1);
  if (p === 'M') return Date.UTC(y, m, 1);
  const dow = (d.getUTCDay() + 6) % 7; // Monday = 0
  return Date.UTC(y, m, d.getUTCDate() - dow);
}
function nextPeriod(ts, p) {
  const d = new Date(ts);
  const y = d.getUTCFullYear(), m = d.getUTCMonth();
  if (p === 'Y') return Date.UTC(y + 1, 0, 1);
  if (p === 'Q') return Date.UTC(y, m + 3, 1);
  if (p === 'M') return Date.UTC(y, m + 1, 1);
  return ts + 7 * 86400000;
}
function periodLabel(ts, p) {
  const d = new Date(ts);
  const y = d.getUTCFullYear();
  if (p === 'Y') return String(y);
  if (p === 'Q') return `Q${Math.floor(d.getUTCMonth() / 3) + 1} ${y}`;
  if (p === 'M') return d.toLocaleDateString('en-GB', { month: 'short', year: '2-digit', timeZone: 'UTC' });
  return d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: '2-digit', timeZone: 'UTC' });
}

function renderTrend(data) {
  const p = state.trendPeriod;
  const dr = state.metrics.date_range;
  const keys = [];
  for (let ts = periodStart(toDate(dr.min), p); ts <= toDate(dr.max).getTime(); ts = nextPeriod(ts, p)) keys.push(ts);
  const index = new Map(keys.map((k, i) => [k, i]));
  const counts = keys.map(() => 0);
  data.forEach(c => { counts[index.get(periodStart(toDate(c.Dt_Customer), p))] += 1; });
  const labels = keys.map(k => periodLabel(k, p));
  const avg = counts.length ? mean(counts, v => v) : 0;
  const last = counts.length - 1;

  // Summary line: latest period vs previous period
  let chg = '';
  if (last >= 1) {
    const prev = counts[last - 1], cur = counts[last];
    if (prev > 0) {
      const pct = (cur - prev) / prev;
      chg = pct >= 0
        ? `<span class="chg-up">▲ ${fmtPct(pct)}</span> vs previous ${PERIOD_NAMES[p]}`
        : `<span class="chg-down">▼ ${fmtPct(Math.abs(pct))}</span> vs previous ${PERIOD_NAMES[p]}`;
    } else {
      chg = `no enrollments in the previous ${PERIOD_NAMES[p]}`;
    }
  }
  $('trend-summary').innerHTML = counts.length
    ? `Latest ${PERIOD_NAMES[p]} (${labels[last]}): <strong>${fmtInt(counts[last])}</strong> new customers ${chg ? `· ${chg}` : ''}`
    : '';

  const showLabels = p !== 'W';
  getChart('chart-trend').setOption({
    textStyle: { fontFamily: FONT },
    grid: { left: 4, right: 36, top: 22, bottom: 4, containLabel: true },
    tooltip: {
      ...TOOLTIP, trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(0,0,0,.03)' } },
      formatter: ps => `<b>${ps[0].name}</b><br/>New customers: <b>${fmtInt(ps[0].value)}</b><br/>Average: ${avg.toFixed(1)}`,
    },
    xAxis: {
      type: 'category', data: labels,
      axisLine: { lineStyle: { color: C.greyLight } }, axisTick: { show: false },
      axisLabel: { color: C.muted, fontSize: 10, hideOverlap: true },
    },
    yAxis: { type: 'value', show: false, splitLine: { show: false } },
    series: [{
      type: 'bar',
      barCategoryGap: p === 'W' ? '20%' : '30%',
      data: counts.map((v, i) => ({
        value: v,
        itemStyle: { color: i === last ? C.orange : C.orangePale, borderRadius: [3, 3, 0, 0] },
        label: { show: showLabels || i === last, position: 'top', color: i === last ? C.orange : C.muted, fontSize: 10, fontWeight: i === last ? 700 : 400 },
      })),
      markLine: {
        symbol: 'none', silent: true,
        lineStyle: { type: 'dashed', color: C.grey, width: 1 },
        label: { position: 'end', formatter: `Avg\n${avg.toFixed(1)}`, color: C.muted, fontSize: 10 },
        data: [{ yAxis: avg }],
      },
    }],
  }, true);
}

/* ---------- Latent space ---------- */
function renderLatent(data) {
  const chart = getChart('chart-latent');
  const point = c => [c.latent_x, c.latent_y, c.ID, c.Income, c.Total_Spending, c.cluster_name];
  const series = [{
    name: 'All customers', type: 'scatter', silent: true, symbolSize: 4,
    itemStyle: { color: C.faint }, data: state.customers.map(point), z: 1,
  }];
  state.segments.forEach(s => {
    series.push({
      name: s.name, type: 'scatter', symbolSize: 6, z: 2,
      itemStyle: { color: state.segColor[s.name], opacity: 0.75 },
      emphasis: { itemStyle: { opacity: 1, borderColor: '#fff', borderWidth: 1 } },
      data: data.filter(c => c.cluster_name === s.name).map(point),
    });
  });
  chart.setOption({
    textStyle: { fontFamily: FONT },
    legend: {
      top: 0, left: 0, icon: 'circle', itemWidth: 9, itemHeight: 9, itemGap: 16,
      textStyle: { color: C.muted, fontSize: 11 }, data: state.segments.map(s => s.name),
    },
    grid: { left: 4, right: 12, top: 26, bottom: 2, containLabel: true },
    tooltip: {
      ...TOOLTIP, trigger: 'item',
      formatter: p => `<b>${custId(p.value[2])}</b><br/>${p.marker}${escapeHtml(p.value[5])}<br/>` +
        `Income: <b>${fmtMoney(p.value[3])}</b><br/>Spending: <b>${fmtMoney(p.value[4])}</b>`,
    },
    xAxis: { type: 'value', axisLabel: { color: C.muted, fontSize: 10 }, splitLine: { lineStyle: { color: '#F3F3F3' } }, axisLine: { show: false }, axisTick: { show: false }, scale: true },
    yAxis: { type: 'value', axisLabel: { color: C.muted, fontSize: 10 }, splitLine: { lineStyle: { color: '#F3F3F3' } }, axisLine: { show: false }, axisTick: { show: false }, scale: true },
    series,
  }, true);
  onceClick(chart, p => { if (p.seriesName !== 'All customers') toggleFilter('cluster_name', p.seriesName); });
}

/* ---------- Static model charts ---------- */
function renderLoss() {
  const h = state.metrics.loss_history;
  const epochs = h.loss.map((_, i) => i + 1);
  $('loss-meta').textContent = `best epoch ${h.best_epoch} · val MSE ${Math.min(...h.val_loss).toFixed(3)}`;
  getChart('chart-loss').setOption({
    textStyle: { fontFamily: FONT },
    legend: { top: 0, right: 0, icon: 'roundRect', itemWidth: 12, itemHeight: 3, textStyle: { color: C.muted, fontSize: 11 } },
    grid: { left: 4, right: 14, top: 24, bottom: 2, containLabel: true },
    tooltip: { ...TOOLTIP, trigger: 'axis', valueFormatter: v => v.toFixed(4) },
    xAxis: { type: 'category', data: epochs, axisLabel: { color: C.muted, fontSize: 10 }, axisTick: { show: false }, axisLine: { lineStyle: { color: C.greyLight } } },
    yAxis: { type: 'value', axisLabel: { color: C.muted, fontSize: 10 }, splitLine: { lineStyle: { color: '#F3F3F3' } }, scale: true },
    series: [
      { name: 'Training', type: 'line', data: h.loss, showSymbol: false, smooth: true, lineStyle: { color: C.slate, width: 2 }, itemStyle: { color: C.slate } },
      {
        name: 'Validation', type: 'line', data: h.val_loss, showSymbol: false, smooth: true,
        lineStyle: { color: C.orange, width: 2 }, itemStyle: { color: C.orange },
        markLine: {
          symbol: 'none', silent: true, lineStyle: { type: 'dashed', color: C.grey },
          label: { show: false },
          data: [{ xAxis: String(h.best_epoch) }],
        },
      },
    ],
  });
}

function renderOptimalK() {
  const m = state.metrics;
  const ks = m.k_metrics.map(r => String(r.k));
  const chosenIdx = ks.indexOf(String(m.chosen_k));
  getChart('chart-k').setOption({
    textStyle: { fontFamily: FONT },
    legend: { top: 0, right: 0, icon: 'roundRect', itemWidth: 12, itemHeight: 3, textStyle: { color: C.muted, fontSize: 11 }, formatter: n => `${n} (${n === 'Inertia' ? 'left' : 'right'})` },
    grid: { left: 4, right: 4, top: 24, bottom: 2, containLabel: true },
    tooltip: { ...TOOLTIP, trigger: 'axis', formatter: ps => `<b>k = ${ps[0].name}</b><br/>` + ps.map(p => `${p.marker}${p.seriesName}: <b>${p.seriesName === 'Inertia' ? fmtInt(p.value) : p.value.toFixed(3)}</b>`).join('<br/>') },
    xAxis: { type: 'category', data: ks, axisLabel: { color: C.muted }, axisTick: { show: false }, axisLine: { lineStyle: { color: C.greyLight } } },
    yAxis: [
      { type: 'value', axisLabel: { color: C.muted, fontSize: 10, formatter: v => (v >= 1000 ? `${(v / 1000).toFixed(0)}k` : v) }, splitLine: { lineStyle: { color: '#F3F3F3' } }, scale: true },
      { type: 'value', axisLabel: { color: C.muted, fontSize: 10, formatter: v => v.toFixed(2) }, splitLine: { show: false }, scale: true },
    ],
    series: [
      { name: 'Inertia', type: 'line', data: m.k_metrics.map(r => r.inertia), symbolSize: 6, lineStyle: { color: C.slate, width: 2 }, itemStyle: { color: C.slate } },
      {
        name: 'Silhouette', type: 'line', yAxisIndex: 1, data: m.k_metrics.map(r => r.silhouette), symbolSize: 6,
        lineStyle: { color: C.orange, width: 2 }, itemStyle: { color: C.orange },
        markPoint: {
          symbol: 'circle', symbolSize: 14, itemStyle: { color: C.orange, borderColor: '#fff', borderWidth: 2 },
          label: { show: true, position: 'top', formatter: `k = ${m.chosen_k}`, color: C.orange, fontWeight: 700, fontSize: 11 },
          data: [{ coord: [chosenIdx, m.k_metrics[chosenIdx].silhouette] }],
        },
        markLine: { symbol: 'none', silent: true, label: { show: false }, lineStyle: { type: 'dashed', color: C.orange2 }, data: [{ xAxis: chosenIdx }] },
      },
    ],
  });
}

function renderComparison() {
  const rows = state.metrics.comparison;
  const cols = [
    { key: 'silhouette', label: 'Silhouette ↑', better: 'max', fmt: v => v.toFixed(3) },
    { key: 'davies_bouldin', label: 'Davies-Bouldin ↓', better: 'min', fmt: v => v.toFixed(3) },
    { key: 'calinski_harabasz', label: 'Calinski-Harabasz ↑', better: 'max', fmt: v => fmtInt(v) },
  ];
  const best = {};
  cols.forEach(c => {
    const vals = rows.map(r => r[c.key]);
    best[c.key] = c.better === 'max' ? Math.max(...vals) : Math.min(...vals);
  });
  $('compare-k').textContent = `k = ${state.metrics.chosen_k}`;
  $('compare-table').innerHTML = `
    <thead><tr><th>Method</th>${cols.map(c => `<th>${c.label}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r => `
      <tr class="${r.proposed ? 'is-proposed' : ''}">
        <td title="Scored in: ${escapeHtml(r.space)}">${escapeHtml(r.method)}</td>
        ${cols.map(c => `<td class="${r[c.key] === best[c.key] ? 'best' : ''}">${c.fmt(r[c.key])}</td>`).join('')}
      </tr>`).join('')}
    </tbody>`;
}

/* ---------- Segment profile (filter-aware) ---------- */
function renderProfile(data) {
  const cols = [
    { label: 'Income', f: s => mean(s, d => d.Income), fmt: fmtMoney },
    { label: 'Spend', f: s => mean(s, d => d.Total_Spending), fmt: fmtMoney },
    { label: 'Kids', f: s => mean(s, d => d.Total_Children), fmt: v => v.toFixed(2) },
    { label: 'Age', f: s => mean(s, d => d.Age), fmt: v => v.toFixed(0) },
    { label: 'Web %', f: s => mean(s, d => (d.Total_Purchases ? d.NumWebPurchases / d.Total_Purchases : 0)), fmt: v => fmtPct(v, 0) },
    { label: 'Resp %', f: s => mean(s, d => d.Response), fmt: v => fmtPct(v, 1) },
  ];
  const rows = state.segments
    .map(seg => ({ seg, subset: data.filter(d => d.cluster_name === seg.name) }))
    .filter(r => r.subset.length)
    .map(r => ({ ...r, vals: cols.map(c => c.f(r.subset)) }));
  if (!rows.length) {
    $('profile-table').innerHTML = '<tbody><tr><td class="muted-empty">No customers match.</td></tr></tbody>';
    return;
  }
  const maxes = cols.map((_, i) => Math.max(...rows.map(r => r.vals[i])));
  $('profile-table').innerHTML = `
    <thead><tr><th>Segment</th><th>n</th>${cols.map(c => `<th>${c.label}</th>`).join('')}</tr></thead>
    <tbody>${rows.map(r => `
      <tr>
        <td><span class="seg-dot" style="background:${state.segColor[r.seg.name]}"></span>${escapeHtml(r.seg.name)}</td>
        <td>${fmtInt(r.subset.length)}</td>
        ${r.vals.map((v, i) => `<td class="${rows.length > 1 && v === maxes[i] ? 'best' : ''}">${cols[i].fmt(v)}</td>`).join('')}
      </tr>`).join('')}
    </tbody>`;
}

/* ---------- Anomaly watch (filter-aware) ---------- */
function renderAnomaly(data) {
  const top = [...data].sort((a, b) => b.recon_error - a.recon_error).slice(0, 10);
  if (!top.length) {
    $('anomaly-list').innerHTML = '<div class="muted-empty">No customers match.</div>';
    return;
  }
  const maxErr = top[0].recon_error || 1;
  $('anomaly-list').innerHTML = top.map((c, i) => `
    <div class="anom" title="Reconstruction error ${c.recon_error.toFixed(3)}">
      <span class="anom__rank">${i + 1}</span>
      <div>
        <div class="anom__id">${custId(c.ID)}</div>
        <div class="anom__sub">${escapeHtml(c.cluster_name)} · ${fmtMoney(c.Income)} income · ${fmtMoney(c.Total_Spending)} spend</div>
      </div>
      <div>
        <div class="anom__bar" style="width:${(c.recon_error / maxErr) * 100}%"></div>
        <div class="anom__err">err ${c.recon_error.toFixed(2)}</div>
      </div>
    </div>`).join('');
}

document.addEventListener('DOMContentLoaded', init);
