/* ==========================================================================
   Dataset page: the complete raw dataset (data/dataset.csv, written by the
   pipeline) in a searchable/sortable table, plus upload / restore actions.
   Relies on helpers and `state` from app.js.
   ========================================================================== */
'use strict';

const ds = {
  loaded: false,
  loading: false,
  columns: [],      // raw CSV column names
  numericCols: [],  // per column: true if (mostly) numeric
  missingByCol: [], // per column: count of empty cells
  rows: [],         // [{ i, cells, cust, missing, search }]
  view: [],
  sortCol: null,    // -1 = segment column, >= 0 = raw column index
  sortDir: 1,
  page: 0,
  pageSize: 50,
  serverAvailable: true,
  pollTimer: null,
};

function initDataset() {
  const m = state.metrics;
  if (m && m.dataset) {
    $('ds-name').textContent = m.dataset.name;
    $('ds-sub').textContent = `${m.dataset.is_original ? 'Original dataset' : 'Uploaded dataset'} · processed ${m.generated_at} · ` +
      `${fmtInt(m.dataset.rows)} rows, ${m.dataset.columns.length} columns · k = ${m.chosen_k} segments`;
    $('ds-restore').hidden = m.dataset.is_original;
    $('ds-download').setAttribute('download', m.dataset.name);
    renderFormatHelp();
  }

  $('ds-upload').addEventListener('click', () => {
    if (!ds.serverAvailable) return showServerHint();
    $('ds-file-input').click();
  });
  $('ds-file-input').addEventListener('change', e => {
    const file = e.target.files[0];
    e.target.value = '';
    if (file) uploadDataset(file);
  });
  $('ds-restore').addEventListener('click', restoreDataset);

  let t;
  $('ds-search').addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => { ds.page = 0; applyDatasetView(); }, 150); });
  $('ds-filter').addEventListener('change', () => { ds.page = 0; applyDatasetView(); });
  $('ds-table').addEventListener('click', onTableClick);
  $('ds-pager').addEventListener('click', onPagerClick);
  $('ds-pager').addEventListener('change', e => {
    if (e.target.id === 'ds-page-size') { ds.pageSize = Number(e.target.value); ds.page = 0; renderDatasetTable(); }
  });

  checkServer();
}

/** Called by the router whenever the Dataset page is shown. */
function showDatasetPage() {
  if (!ds.loaded && !ds.loading && state.metrics) loadDataset();
}

/* ---------- Loading & indexing ---------- */
function loadDataset() {
  ds.loading = true;
  $('ds-count').textContent = 'Loading dataset…';
  const bust = encodeURIComponent(state.metrics.generated_at || Date.now());
  Papa.parse(`data/dataset.csv?v=${bust}`, {
    download: true,
    header: true,
    skipEmptyLines: 'greedy',
    transformHeader: h => h.trim(),
    complete: res => { ds.loading = false; indexDataset(res); },
    error: err => {
      ds.loading = false;
      $('ds-count').textContent = `Could not load the dataset file (${err.message || err}).`;
    },
  });
}

function indexDataset(res) {
  const columns = res.meta.fields.filter(c => c !== '');
  const byRow = new Map(state.customers.map(c => [c.row, c]));
  ds.columns = columns;
  ds.missingByCol = columns.map(() => 0);
  const numericCount = columns.map(() => 0);

  ds.rows = res.data.map((r, i) => {
    const cells = columns.map(c => (r[c] == null ? '' : String(r[c]).trim()));
    let missing = false;
    cells.forEach((v, j) => {
      if (v === '') { ds.missingByCol[j] += 1; missing = true; } else if (!Number.isNaN(Number(v))) numericCount[j] += 1;
    });
    const cust = byRow.get(i) || null;
    return { i, cells, cust, missing, search: `${cells.join(' ')} ${cust ? cust.cluster_name : 'excluded'}`.toLowerCase() };
  });
  ds.numericCols = columns.map((_, j) => numericCount[j] >= 0.8 * (ds.rows.length - ds.missingByCol[j]));
  ds.loaded = true;
  renderDatasetStats();
  applyDatasetView();
}

/* ---------- Stats tiles ---------- */
function renderDatasetStats() {
  const totalMissing = ds.missingByCol.reduce((a, b) => a + b, 0);
  const colsWithMissing = ds.missingByCol.filter(v => v > 0).length;
  const included = ds.rows.filter(r => r.cust).length;
  const dr = state.metrics.date_range;
  const tiles = [
    [fmtInt(ds.rows.length), 'Total rows', false],
    [fmtInt(ds.columns.length), 'Columns', false],
    [fmtInt(totalMissing), `Missing values${colsWithMissing ? ` (${colsWithMissing} col.)` : ''}`, totalMissing > 0],
    [fmtInt(included), 'Customers in analysis', true],
    [fmtInt(ds.rows.length - included), 'Excluded outliers', false],
    [`${fmtDate(dr.min)} – ${fmtDate(dr.max)}`, 'Enrollment range', false],
  ];
  $('ds-stats').innerHTML = tiles.map(([v, k, accent]) => `
    <div class="stat-tile">
      <div class="stat-tile__v ${accent ? 'stat-tile__v--accent' : ''}" ${v.length > 14 ? 'style="font-size:16px;padding-top:6px"' : ''}>${escapeHtml(v)}</div>
      <div class="stat-tile__k">${escapeHtml(k)}</div>
    </div>`).join('');
}

function renderFormatHelp() {
  const m = state.metrics;
  $('ds-required').innerHTML = (m.required_columns || []).map(c => `<code>${escapeHtml(c)}</code>`).join('');
  $('ds-optional').innerHTML = (m.optional_columns || []).map(c => `<code>${escapeHtml(c)}</code>`).join('');
}

/* ---------- Filtering, sorting, paging ---------- */
function applyDatasetView() {
  if (!ds.loaded) return;
  const q = $('ds-search').value.trim().toLowerCase();
  const mode = $('ds-filter').value;
  let view = ds.rows.filter(r => {
    if (mode === 'included' && !r.cust) return false;
    if (mode === 'excluded' && r.cust) return false;
    if (mode === 'missing' && !r.missing) return false;
    return !q || r.search.includes(q);
  });
  if (ds.sortCol !== null) {
    const col = ds.sortCol;
    const numeric = col >= 0 && ds.numericCols[col];
    const key = r => (col === -1 ? (r.cust ? r.cust.cluster_name : '~') : r.cells[col]);
    view = [...view].sort((a, b) => {
      const va = key(a), vb = key(b);
      if (va === '' && vb !== '') return 1;   // blanks always last
      if (vb === '' && va !== '') return -1;
      const cmp = numeric ? Number(va) - Number(vb) : String(va).localeCompare(String(vb), undefined, { numeric: true });
      return cmp * ds.sortDir;
    });
  }
  ds.view = view;
  renderDatasetTable();
}

function renderDatasetTable() {
  const total = ds.view.length;
  const pages = Math.max(1, Math.ceil(total / ds.pageSize));
  ds.page = Math.min(ds.page, pages - 1);
  const start = ds.page * ds.pageSize;
  const slice = ds.view.slice(start, start + ds.pageSize);
  $('ds-count').textContent = `${fmtInt(total)} of ${fmtInt(ds.rows.length)} rows`;

  const arrow = c => (ds.sortCol === c ? (ds.sortDir === 1 ? ' ▲' : ' ▼') : '');
  const head = `<thead><tr>
      <th data-col="row">#</th>
      <th data-col="-1" class="${ds.sortCol === -1 ? 'sorted' : ''}">Segment${arrow(-1)}</th>
      ${ds.columns.map((c, j) => `<th data-col="${j}" class="${ds.sortCol === j ? 'sorted' : ''}" title="${ds.missingByCol[j]} missing value(s)">${escapeHtml(c)}${arrow(j)}${ds.missingByCol[j] ? `<span class="miss-badge">${ds.missingByCol[j]}</span>` : ''}</th>`).join('')}
    </tr></thead>`;
  const body = slice.length
    ? slice.map(r => `<tr data-row="${r.i}" class="${r.cust ? '' : 'excluded'}" title="${r.cust ? `Analyse ${custId(r.cust.ID)}` : 'Excluded from the analysis as an outlier'}">
        <td class="rownum">${r.i + 1}</td>
        <td>${r.cust ? `<span class="seg-dot" style="background:${state.segColor[r.cust.cluster_name]}"></span>${escapeHtml(r.cust.cluster_name)}` : '<span class="tag">Excluded</span>'}</td>
        ${r.cells.map((v, j) => (v === '' ? '<td class="miss">—</td>' : `<td class="${ds.numericCols[j] ? 'num' : ''}">${escapeHtml(v)}</td>`)).join('')}
      </tr>`).join('')
    : `<tr><td colspan="${ds.columns.length + 2}" class="muted-empty">No rows match.</td></tr>`;
  $('ds-table').innerHTML = head + `<tbody>${body}</tbody>`;

  $('ds-pager').innerHTML = `
    <button class="btn" data-page="first" ${ds.page === 0 ? 'disabled' : ''}>«</button>
    <button class="btn" data-page="prev" ${ds.page === 0 ? 'disabled' : ''}>‹ Prev</button>
    <span class="ds-pager__info">Page <strong>${ds.page + 1}</strong> of ${pages} · rows ${total ? start + 1 : 0}–${Math.min(start + ds.pageSize, total)}</span>
    <button class="btn" data-page="next" ${ds.page >= pages - 1 ? 'disabled' : ''}>Next ›</button>
    <button class="btn" data-page="last" ${ds.page >= pages - 1 ? 'disabled' : ''}>»</button>
    <label class="ds-pager__info" style="margin-left:auto">Rows per page
      <select id="ds-page-size">${[25, 50, 100, 250].map(n => `<option value="${n}" ${n === ds.pageSize ? 'selected' : ''}>${n}</option>`).join('')}</select>
    </label>`;
}

function onTableClick(e) {
  const th = e.target.closest('th');
  if (th) {
    if (th.dataset.col === 'row') { ds.sortCol = null; ds.sortDir = 1; }
    else {
      const col = Number(th.dataset.col);
      if (ds.sortCol === col) ds.sortDir *= -1; else { ds.sortCol = col; ds.sortDir = 1; }
    }
    applyDatasetView();
    return;
  }
  const tr = e.target.closest('tr[data-row]');
  if (!tr) return;
  const row = ds.rows[Number(tr.dataset.row)];
  if (row && row.cust) navigate('ai', `id=${row.cust.ID}`);
  else toast('This row was excluded from the analysis as an outlier (Age > 100 or Income > 600k).', 'error', 4000);
}

function onPagerClick(e) {
  const btn = e.target.closest('button[data-page]');
  if (!btn) return;
  const pages = Math.max(1, Math.ceil(ds.view.length / ds.pageSize));
  const act = btn.dataset.page;
  ds.page = act === 'first' ? 0 : act === 'last' ? pages - 1 : act === 'prev' ? ds.page - 1 : ds.page + 1;
  ds.page = Math.max(0, Math.min(pages - 1, ds.page));
  renderDatasetTable();
  $('ds-table-wrap').scrollTop = 0;
}

/* ---------- Upload / restore ---------- */
async function checkServer() {
  try {
    const res = await fetch('api/status', { cache: 'no-store' });
    if (!res.ok) throw new Error(String(res.status));
    const st = await res.json();
    if (st.state === 'running') { showUploadPanel(); pollStatus(); }
  } catch {
    ds.serverAvailable = false;
    $('ds-upload').title = 'Start the app with "python server.py" to enable uploads';
  }
}

function showServerHint() {
  showUploadPanel();
  setUploadState('error', 'Uploads need the app server',
    'This page is being served by a plain static server. Stop it and run "python server.py" from the ' +
    'customer-dashboard folder, then open http://localhost:8000 again.', 0);
}

async function uploadDataset(file) {
  const ok = window.confirm(
    `Replace the current dataset with "${file.name}"?\n\n` +
    'The autoencoder and K-Means models will be retrained on this file (about 30 seconds), ' +
    'and the Dashboard, Dataset and AI Analysis pages will update to the new data.');
  if (!ok) return;

  showUploadPanel();
  setUploadState('running', `Uploading ${file.name}…`, 'Sending file to the server', 1);
  let res, body = {};
  try {
    res = await fetch('api/upload', {
      method: 'POST',
      headers: { 'X-Filename': encodeURIComponent(file.name), 'Content-Type': 'application/octet-stream' },
      body: file,
    });
    body = await res.json().catch(() => ({}));
  } catch (err) {
    return setUploadState('error', 'Upload failed', `Could not reach the server (${err.message}).`, 0);
  }
  if (res.status === 404 || res.status === 501) { ds.serverAvailable = false; return showServerHint(); }
  if (!res.ok) return setUploadState('error', 'File rejected', body.error || `Server returned HTTP ${res.status}.`, 0);
  setUploadState('running', `Processing ${file.name}`, `Accepted ${fmtInt(body.rows)} rows × ${body.columns} columns — starting pipeline`, 3);
  pollStatus();
}

async function restoreDataset() {
  if (!window.confirm('Restore the original dataset? The models will be retrained on it (about 30 seconds).')) return;
  showUploadPanel();
  setUploadState('running', 'Restoring original dataset', 'Starting pipeline', 1);
  try {
    const res = await fetch('api/restore', { method: 'POST' });
    if (res.status === 404 || res.status === 501) return showServerHint();
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      return setUploadState('error', 'Could not restore', body.error || `HTTP ${res.status}`, 0);
    }
  } catch (err) {
    return setUploadState('error', 'Could not restore', err.message, 0);
  }
  pollStatus();
}

function pollStatus() {
  clearTimeout(ds.pollTimer);
  $('ds-upload').disabled = true;
  $('ds-restore').disabled = true;
  const tick = async () => {
    let st;
    try {
      st = await (await fetch('api/status', { cache: 'no-store' })).json();
    } catch {
      ds.pollTimer = setTimeout(tick, 2000);
      return;
    }
    $('upload-log').textContent = (st.log || []).join('\n');
    $('upload-log').scrollTop = $('upload-log').scrollHeight;
    if (st.state === 'running') {
      setUploadState('running', `Processing ${st.dataset}`, st.stage, st.progress);
      ds.pollTimer = setTimeout(tick, 1000);
    } else if (st.state === 'done') {
      setUploadState('done', 'Done — reloading the dashboard…', st.message, 100);
      try { sessionStorage.setItem('pendingToast', st.message); } catch { /* ignore */ }
      setTimeout(() => { location.hash = '#/dashboard'; location.reload(); }, 900);
    } else if (st.state === 'error') {
      setUploadState('error', 'Pipeline failed — the previous data is still in place', st.message, st.progress);
      $('ds-upload').disabled = false;
      $('ds-restore').disabled = false;
    }
  };
  tick();
}

function showUploadPanel() {
  $('upload-panel').hidden = false;
  $('upload-log').textContent = '';
}

function setUploadState(kind, title, stage, pct) {
  const panel = $('upload-panel');
  panel.classList.toggle('upload--error', kind === 'error');
  panel.classList.toggle('upload--done', kind === 'done');
  $('upload-title').textContent = title;
  $('upload-stage').textContent = stage || '';
  $('upload-pct').textContent = kind === 'error' ? 'Failed' : `${Math.round(pct)}%`;
  $('upload-fill').style.width = `${kind === 'error' ? 100 : pct}%`;
  if (kind === 'error') { $('ds-upload').disabled = false; $('ds-restore').disabled = false; }
}
