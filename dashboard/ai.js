/* ==========================================================================
   AI Analysis page: pick one customer and explain the model's view of them.

   Everything comes from the trained models (via customers.json):
   - segment = K-Means cluster of the customer's 8-D autoencoder embedding
   - seg_probs = soft membership from latent distances to every centroid
   - resp_prob / drivers = logistic-regression response model (out-of-fold)
   - similar = nearest neighbours in the latent space
   - recon_pct = how unusual the customer is for the autoencoder
   The narrative is generated from these numbers and the segment/overall averages.
   Relies on helpers and `state` from app.js.
   ========================================================================== */
'use strict';

const ai = {
  byId: new Map(),
  segStats: {},   // cluster name -> averages
  overall: null,  // averages over all customers
  current: null,
  charts: [],
  listLimit: 300,
};

const AI_METRICS = [
  { key: 'Income', label: 'Income', fmt: v => fmtMoney(v) },
  { key: 'Total_Spending', label: 'Total spend', fmt: v => fmtMoney(v) },
  { key: 'Total_Purchases', label: 'Purchases', fmt: v => v.toFixed(1) },
  { key: 'NumDealsPurchases', label: 'Deal purchases', fmt: v => v.toFixed(1) },
  { key: 'NumWebVisitsMonth', label: 'Web visits / month', fmt: v => v.toFixed(1) },
  { key: 'Recency', label: 'Days since purchase', fmt: v => v.toFixed(0) },
  { key: 'Total_Children', label: 'Children', fmt: v => v.toFixed(2) },
  { key: 'Age', label: 'Age', fmt: v => v.toFixed(0) },
  { key: 'Total_Campaigns_Accepted', label: 'Campaigns accepted', fmt: v => v.toFixed(2) },
  { key: 'Customer_Tenure_Days', label: 'Tenure (days)', fmt: v => v.toFixed(0) },
];
const CHANNELS = [
  ['Web', 'NumWebPurchases'], ['Catalog', 'NumCatalogPurchases'], ['Store', 'NumStorePurchases'],
  ['Deals', 'NumDealsPurchases'], ['Web visits', 'NumWebVisitsMonth'],
];

const FEATURE_LABELS = {
  Age: 'Age', Income: 'Income', Kidhome: 'Young children at home', Teenhome: 'Teenagers at home',
  Recency: 'Days since last purchase', MntWines: 'Wine spend', MntFruits: 'Fruit spend',
  MntMeatProducts: 'Meat spend', MntFishProducts: 'Fish spend', MntSweetProducts: 'Sweets spend',
  MntGoldProds: 'Gold spend', NumDealsPurchases: 'Deal purchases', NumWebPurchases: 'Web purchases',
  NumCatalogPurchases: 'Catalog purchases', NumStorePurchases: 'Store purchases',
  NumWebVisitsMonth: 'Web visits per month', Complain: 'Complaint filed', Total_Spending: 'Total spend',
  Total_Purchases: 'Total purchases', Total_Children: 'Children', Has_Partner: 'Has a partner',
  Family_Size: 'Family size', Customer_Tenure_Days: 'Customer tenure', Total_Campaigns_Accepted: 'Past campaigns accepted',
};
const featureLabel = f => FEATURE_LABELS[f]
  || (f.startsWith('AcceptedCmp') ? `Accepted campaign ${f.slice(-1)}` : null)
  || (f.startsWith('Edu_') ? `Education: ${f.slice(4)}` : null)
  || (f.startsWith('Marital_') ? `Marital status: ${f.slice(8)}` : null)
  || f.replace(/_/g, ' ');

/* Segment playbooks, keyed by the base persona name (without "II", "III" suffixes). */
const PLAYBOOK = {
  'Premium Shoppers': [
    'Offer premium bundles (wine + meat) and early access to new products rather than discounts.',
    'Use a loyalty tier or VIP events; they buy across catalog, store and web.',
    'Protect margin: deals are not needed to win this group.',
  ],
  'Campaign Enthusiasts': [
    'Include in every campaign: this group has by far the highest response rate.',
    'Pilot new offers here first; it is the best test audience for campaign creative.',
    'Reward each response with exclusive perks to keep engagement high.',
  ],
  'Mainstream Customers': [
    'They spend well but ignore campaigns, so test new channels, timing and messages.',
    'Personalise recommendations around their main product category.',
    'Use moderate, convenience-led incentives instead of heavy discounting.',
  ],
  'Budget Families': [
    'Lead with deals, coupons and family-size value bundles.',
    'Avoid costly premium (wine / gold) catalogue mailings; expected return is low.',
    'Use web retargeting: they browse often but buy little online.',
  ],
  'Digital-First Buyers': [
    'Prioritise email and web retargeting; offer web-only promotions.',
    'Use free-delivery thresholds to grow basket size online.',
    'Keep the online journey fast; this group converts on the web.',
  ],
};
const basePersona = name => name.replace(/\s+(II|III|IV|V|VI|\d+)$/, '');

/* ---------- Setup ---------- */
function initAI() {
  ai.byId = new Map(state.customers.map(c => [String(c.ID), c]));
  ai.overall = averages(state.customers);
  state.segments.forEach(s => { ai.segStats[s.name] = averages(state.customers.filter(c => c.cluster_name === s.name)); });

  $('ai-seg-filter').innerHTML = '<option value="">All segments</option>' +
    state.segments.map(s => `<option value="${escapeHtml(s.name)}">${escapeHtml(s.name)}</option>`).join('');
  let t;
  $('ai-search').addEventListener('input', () => { clearTimeout(t); t = setTimeout(renderAIList, 120); });
  $('ai-seg-filter').addEventListener('change', renderAIList);
  $('ai-random').addEventListener('click', () => {
    const pool = filteredAIList();
    const pick = pool[Math.floor(Math.random() * pool.length)];
    if (pick) navigate('ai', `id=${pick.ID}`);
  });
  $('ai-list').addEventListener('click', e => {
    const item = e.target.closest('[data-id]');
    if (item) navigate('ai', `id=${item.dataset.id}`);
  });
  $('ai-result').addEventListener('click', e => {
    const item = e.target.closest('[data-goto]');
    if (item) navigate('ai', `id=${item.dataset.goto}`);
  });
  renderAIList();
}

function averages(list) {
  const out = { n: list.length };
  const keys = [...AI_METRICS.map(m => m.key), ...MNT_KEYS(), ...CHANNELS.map(c => c[1]), 'Response', 'resp_prob'];
  keys.forEach(k => { out[k] = mean(list, c => Number(c[k]) || 0); });
  out.web_share = mean(list, c => (c.Total_Purchases ? c.NumWebPurchases / c.Total_Purchases : 0));
  return out;
}
function MNT_KEYS() { return PRODUCTS.map(p => p[1]); }

/** Called by the router whenever the AI page is shown. */
function showAIPage(id) {
  if (!state.customers.length) return;
  const c = id ? ai.byId.get(String(id).replace(/^c_?/i, '')) : null;
  if (!c) {
    ai.current = null;
    $('ai-empty').hidden = false;
    $('ai-result').hidden = true;
    if (id) toast(`Customer ${id} was not found in the analysed data.`, 'error', 3500);
    highlightAIList();
    return;
  }
  if (ai.current === c) return;
  ai.current = c;
  $('ai-empty').hidden = true;
  $('ai-result').hidden = false;
  renderAnalysis(c);
  highlightAIList(true);
  $('page-ai').scrollTop = 0;
  window.scrollTo(0, 0);
}

/* ---------- Customer picker ---------- */
function filteredAIList() {
  const q = $('ai-search').value.trim().replace(/^c_?/i, '');
  const seg = $('ai-seg-filter').value;
  return state.customers.filter(c => (!seg || c.cluster_name === seg) && (!q || String(c.ID).includes(q)));
}

function renderAIList() {
  const list = filteredAIList();
  const shown = list.slice(0, ai.listLimit);
  $('ai-count').textContent = list.length > shown.length
    ? `Showing ${fmtInt(shown.length)} of ${fmtInt(list.length)} customers. Type an ID to narrow down.`
    : `${fmtInt(list.length)} customer${list.length === 1 ? '' : 's'}`;
  $('ai-list').innerHTML = shown.length
    ? shown.map(c => `
      <div class="ai-item" data-id="${c.ID}">
        <div class="ai-item__id">${custId(c.ID)}</div>
        <div class="ai-item__sub"><span class="seg-dot" style="background:${state.segColor[c.cluster_name]}"></span>${escapeHtml(c.cluster_name)} · ${fmtMoney(c.Income)} · spend ${fmtMoney(c.Total_Spending)}</div>
      </div>`).join('')
    : '<div class="muted-empty">No customers match.</div>';
  highlightAIList();
}

function highlightAIList(scroll = false) {
  document.querySelectorAll('.ai-item').forEach(el => {
    const on = ai.current && el.dataset.id === String(ai.current.ID);
    el.classList.toggle('active', !!on);
    if (on && scroll) el.scrollIntoView({ block: 'nearest' });
  });
}

/* ---------- Narrative helpers ---------- */
function ratioPhrase(value, avg) {
  if (!avg) return 'in line with the average';
  const r = value / avg;
  if (r >= 1.15) return `${r.toFixed(1)}× the average customer`;
  if (r <= 0.85) return `${Math.round((1 - r) * 100)}% below the average customer`;
  return 'close to the average customer';
}

function spendingMix(c) {
  const total = c.Total_Spending || 0;
  return PRODUCTS.map(([name, key]) => ({ name, key, value: c[key], share: total ? c[key] / total : 0 }))
    .sort((a, b) => b.value - a.value);
}

function topChannel(c) {
  const ch = [['the store', c.NumStorePurchases], ['the website', c.NumWebPurchases], ['the catalog', c.NumCatalogPurchases]]
    .sort((a, b) => b[1] - a[1]);
  return { name: ch[0][0], share: c.Total_Purchases ? ch[0][1] / c.Total_Purchases : 0 };
}

function traitsFor(c) {
  const o = ai.overall;
  const traits = [];
  const mix = spendingMix(c);
  const dealShare = c.Total_Purchases ? c.NumDealsPurchases / c.Total_Purchases : 0;
  if (c.Total_Spending >= 2 * o.Total_Spending) traits.push(['High spender', true]);
  else if (c.Total_Spending <= 0.3 * o.Total_Spending) traits.push(['Low spender', false]);
  if (mix[0].share >= 0.5 && c.Total_Spending > 0) traits.push([`${mix[0].name.replace(/s$/, '')} lover`, true]);
  if (dealShare >= 0.3) traits.push(['Deal seeker', true]);
  const tc = topChannel(c);
  if (tc.share >= 0.5) traits.push([`${tc.name.replace('the ', '').replace(/^./, s => s.toUpperCase())} shopper`, false]);
  if (c.NumWebVisitsMonth >= 7 && c.NumWebPurchases <= 3) traits.push(['Browses, rarely buys online', false]);
  if (c.Total_Children >= 2) traits.push(['Large family', false]);
  if (c.Total_Children === 0) traits.push(['No children', false]);
  if (c.Recency <= 15) traits.push(['Recently active', true]);
  if (c.Recency >= 75) traits.push(['Lapsing', false]);
  if (c.Total_Campaigns_Accepted >= 2) traits.push(['Campaign responder', true]);
  if (c.Complain) traits.push(['Has complained', false]);
  if (c.recon_pct >= 95) traits.push(['Unusual profile', true]);
  return traits;
}

function buildSummary(c) {
  const o = ai.overall;
  const kids = c.Total_Children === 0 ? 'no children' : `${c.Total_Children} child${c.Total_Children > 1 ? 'ren' : ''}`;
  const partner = c.Marital_Status === 'Couple' ? 'in a couple' : 'single';
  const months = Math.max(1, Math.round(c.Customer_Tenure_Days / 30.4));
  const mix = spendingMix(c);
  const tc = topChannel(c);
  const probs = c.seg_probs || [];
  const ranked = state.segments.map(s => ({ name: s.name, p: probs[s.id] ?? 0 })).sort((a, b) => b.p - a.p);
  const conf = ranked[0] ? ranked[0].p : 1;
  const runnerUp = ranked[1];
  const segAvg = ai.segStats[c.cluster_name];
  const p = [];

  p.push(`<strong>${custId(c.ID)}</strong> is a ${c.Age}-year-old ${c.Education === 'PhD' ? 'PhD' : c.Education.toLowerCase()} customer who is ${partner}, with ${kids} at home and an income of <strong>${fmtMoney(c.Income)}</strong> (${ratioPhrase(c.Income, o.Income)}).`);

  let spend = `In ${months} months as a customer they have spent <strong>${fmtMoney(c.Total_Spending)}</strong>, ${ratioPhrase(c.Total_Spending, o.Total_Spending)}.`;
  if (c.Total_Spending > 0) {
    spend += ` Their biggest category is <strong>${mix[0].name.toLowerCase()}</strong> (${fmtPct(mix[0].share, 0)} of spend)` +
      (mix[1] && mix[1].value > 0 ? `, followed by ${mix[1].name.toLowerCase()} (${fmtPct(mix[1].share, 0)}).` : '.');
  }
  if (c.Total_Purchases > 0) {
    spend += ` They buy mostly through ${tc.name} (${fmtPct(tc.share, 0)} of ${c.Total_Purchases} purchases)`;
    const dealShare = c.NumDealsPurchases / c.Total_Purchases;
    spend += dealShare >= 0.3 ? `, and ${c.NumDealsPurchases} purchases used a discount, so they are price-sensitive.` : '.';
  }
  p.push(spend);

  let verdict = `The autoencoder compresses their ${state.metrics.n_features} features into an ${state.metrics.latent_dim}-D profile that lies closest to the <strong>${escapeHtml(c.cluster_name)}</strong> centroid. They are classified in that segment with <strong>${fmtPct(conf, 0)}</strong> confidence.`;
  if (runnerUp && runnerUp.p >= 0.2) verdict += ` They also share traits with ${escapeHtml(runnerUp.name)} (${fmtPct(runnerUp.p, 0)}), so they sit near the boundary between the two.`;
  else verdict += ' This is a clear-cut assignment.';
  p.push(verdict);

  if (c.resp_prob != null) {
    const vs = c.resp_prob >= segAvg.resp_prob * 1.2 ? 'above' : c.resp_prob <= segAvg.resp_prob * 0.8 ? 'below' : 'in line with';
    p.push(`The response model estimates a <strong>${fmtPct(c.resp_prob, 0)}</strong> chance that they accept the next campaign, ${vs} the ${fmtPct(segAvg.resp_prob, 0)} average for their segment. ` +
      `They have accepted ${c.Total_Campaigns_Accepted} of 5 earlier campaigns${c.Response ? ' and <strong>responded to the latest one</strong>' : ' and did not respond to the latest one'}.`);
  }
  if (c.recon_pct >= 95) {
    p.push(`<strong>Unusual profile:</strong> the autoencoder struggles to reconstruct this customer (more unusual than ${fmtPct(c.recon_pct / 100, 0)} of customers). Check the record for data errors, or treat them as a niche case.`);
  }
  return p.map(t => `<p>${t}</p>`).join('');
}

function specificActions(c) {
  const out = [];
  const dealShare = c.Total_Purchases ? c.NumDealsPurchases / c.Total_Purchases : 0;
  const mix = spendingMix(c);
  if (c.Complain) out.push('They filed a complaint: follow up with customer service before any marketing.');
  if (c.Recency >= 60) out.push(`Win-back: last purchase was ${c.Recency} days ago. Send a reactivation offer.`);
  else if (c.Recency <= 15) out.push(`Recently active (${c.Recency} days ago): a good moment for a cross-sell.`);
  if (c.resp_prob != null && c.resp_prob >= 0.5) out.push(`High response likelihood (${fmtPct(c.resp_prob, 0)}): include them in the next campaign.`);
  if (c.resp_prob != null && c.resp_prob < 0.08) out.push(`Low response likelihood (${fmtPct(c.resp_prob, 0)}): leave them out of costly mailings to save budget.`);
  if (dealShare >= 0.3) out.push('Deal-sensitive: include a coupon or bundle discount.');
  if (c.NumWebVisitsMonth >= 7 && c.NumWebPurchases <= 3) out.push(`Visits the website ${c.NumWebVisitsMonth}× a month but rarely buys online: retarget these visits.`);
  if (c.Total_Spending > 0 && mix[0].share >= 0.5) out.push(`Spending is concentrated on ${mix[0].name.toLowerCase()} (${fmtPct(mix[0].share, 0)}), so feature it in offers.`);
  return out;
}

/* ---------- Charts ---------- */
function aiChart(el, option) {
  const chart = echarts.init(el);
  chart.setOption(option);
  const ro = new ResizeObserver(() => requestAnimationFrame(() => chart.resize()));
  ro.observe(el);
  ai.charts.push({ chart, ro });
  return chart;
}
function disposeAICharts() {
  ai.charts.forEach(({ chart, ro }) => { ro.disconnect(); chart.dispose(); });
  ai.charts = [];
}

const axisMuted = { axisLabel: { color: C.muted, fontSize: 10 }, axisTick: { show: false }, axisLine: { lineStyle: { color: C.greyLight } } };

function renderAnalysis(c) {
  disposeAICharts();
  const seg = c.cluster_name;
  const segColor = state.segColor[seg];
  const segAvg = ai.segStats[seg];
  const o = ai.overall;
  const probs = c.seg_probs || [];
  const ranked = state.segments.map(s => ({ ...s, p: probs[s.id] ?? 0 })).sort((a, b) => b.p - a.p);
  const prop = state.metrics.propensity || {};
  const traits = traitsFor(c);
  const playbook = PLAYBOOK[basePersona(seg)] || ['Tailor offers to this segment\'s main category and preferred channel.'];
  const specific = specificActions(c);
  const similar = (c.similar || []).map(([id, match]) => ({ c: ai.byId.get(String(id)), match })).filter(s => s.c);
  const maxDriver = Math.max(...(c.drivers || []).map(d => Math.abs(d[1])), 1e-9);

  $('ai-result').innerHTML = `
  <div class="ai-grid">
    <section class="card span-5">
      <div class="verdict__id">${custId(c.ID)}
        ${c.Response ? '<span class="pill pill--orange">Responded to last campaign</span>' : '<span class="pill">Did not respond to last campaign</span>'}
      </div>
      <div class="verdict__profile">${c.Age} yrs · ${escapeHtml(c.Education)} · ${escapeHtml(c.Marital_Status)} · ${c.Total_Children} child${c.Total_Children === 1 ? '' : 'ren'} · customer since ${fmtDate(c.Dt_Customer)}</div>
      <div class="verdict__label">Predicted category</div>
      <div class="verdict__seg" style="color:${segColor}">${escapeHtml(seg)}</div>
      <div class="verdict__conf"><strong>${fmtPct(ranked[0] ? ranked[0].p : 1, 0)}</strong> confidence · Autoencoder (${state.metrics.latent_dim}-D) + K-Means (k = ${state.metrics.chosen_k})</div>
      <div class="verdict__label" style="margin-bottom:6px">Membership across segments</div>
      <div class="memb">${ranked.map(s => `
        <div class="memb__row">
          <span><span class="seg-dot" style="background:${state.segColor[s.name]}"></span>${escapeHtml(s.name)}</span>
          <div class="memb__track"><div class="memb__fill" style="width:${s.p * 100}%;background:${state.segColor[s.name]}"></div></div>
          <span class="memb__pct">${fmtPct(s.p, 0)}</span>
        </div>`).join('')}
      </div>
    </section>

    <section class="card span-7 summary">
      <div class="card__head"><h2 class="card__title">AI Summary</h2><span class="card__meta">Generated from the model outputs</span></div>
      ${buildSummary(c)}
      <div class="traits">${traits.map(([t, hot]) => `<span class="trait ${hot ? 'trait--orange' : ''}">${escapeHtml(t)}</span>`).join('')}</div>
    </section>

    <section class="card span-4">
      <div class="card__head"><h2 class="card__title">Response Propensity</h2></div>
      ${c.resp_prob != null ? `
        <div class="ai-chart ai-chart--gauge" id="ai-gauge"></div>
        <div class="prop__facts">
          <span class="k">Segment average</span><span class="v">${fmtPct(segAvg.resp_prob, 1)}</span>
          <span class="k">All customers</span><span class="v">${fmtPct(o.resp_prob, 1)}</span>
          <span class="k">Model</span><span class="v">Logistic regression · AUC ${prop.auc ? prop.auc.toFixed(3) : '–'}</span>
        </div>
        <div class="verdict__label" style="margin-bottom:6px">What drives this score</div>
        <div class="drivers">${(c.drivers || []).map(([f, v]) => `
          <div class="driver" title="Contribution to the log-odds: ${v >= 0 ? '+' : ''}${v.toFixed(2)}">
            <span class="${v >= 0 ? 'driver__arrow--up' : 'driver__arrow--down'}">${v >= 0 ? '↑' : '↓'}</span>
            <span>${escapeHtml(featureLabel(f))}${c[f] != null && typeof c[f] === 'number' ? ` <span class="muted">(${fmtInt(c[f])})</span>` : ''}</span>
            <div class="driver__bar" style="width:${Math.abs(v) / maxDriver * 100}%;background:${v >= 0 ? C.orange : C.slate}"></div>
          </div>`).join('')}
        </div>` : '<p class="muted">The response model is not available for this dataset (it needs both responders and non-responders).</p>'}
    </section>

    <section class="card span-8">
      <div class="card__head"><h2 class="card__title">Customer vs Segment vs All Customers</h2><span class="card__meta">Index: all-customer average = 100</span></div>
      <div class="ai-chart" id="ai-compare" style="height:330px"></div>
    </section>

    <section class="card span-4">
      <div class="card__head"><h2 class="card__title">Spending Mix</h2><span class="card__meta">${fmtMoney(c.Total_Spending)} total</span></div>
      <div class="ai-chart" id="ai-spend"></div>
    </section>

    <section class="card span-4">
      <div class="card__head"><h2 class="card__title">Channel Behaviour</h2><span class="card__meta">vs segment average</span></div>
      <div class="ai-chart" id="ai-channel"></div>
    </section>

    <section class="card span-4">
      <div class="card__head"><h2 class="card__title">Position in Latent Space</h2></div>
      <div class="ai-chart" id="ai-latent"></div>
    </section>

    <section class="card span-6">
      <div class="card__head"><h2 class="card__title">Most Similar Customers</h2><span class="card__meta">Nearest neighbours in the ${state.metrics.latent_dim}-D latent space</span></div>
      <div class="sim">${similar.length ? similar.map(s => `
        <div class="sim__row" data-goto="${s.c.ID}" title="Analyse ${custId(s.c.ID)}">
          <span class="sim__id">${custId(s.c.ID)}</span>
          <span><span class="seg-dot" style="background:${state.segColor[s.c.cluster_name]}"></span>${escapeHtml(s.c.cluster_name)}</span>
          <span>${fmtMoney(s.c.Income)} income</span>
          <span>${fmtMoney(s.c.Total_Spending)} spend</span>
          <span class="sim__match"><span class="sim__bar" style="width:${s.match * 60}px"></span>${fmtPct(s.match, 0)} match</span>
        </div>`).join('') : '<p class="muted">No neighbours available.</p>'}
      </div>
    </section>

    <section class="card span-6">
      <div class="card__head"><h2 class="card__title">Recommended Actions</h2></div>
      <ul class="actions">
        ${specific.map(a => `<li class="specific">${escapeHtml(a)}</li>`).join('')}
        ${playbook.map(a => `<li>${escapeHtml(a)}</li>`).join('')}
      </ul>
      <div class="anomaly-box ${c.recon_pct >= 95 ? 'anomaly-box--warn' : ''}">
        <strong>Anomaly check:</strong> reconstruction error ${c.recon_error.toFixed(3)}, higher than ${fmtPct(c.recon_pct / 100, 0)} of customers.
        ${c.recon_pct >= 95 ? 'This customer is <strong>unusual</strong> for the model; review the record.' : 'This is a typical profile that the model represents well.'}
      </div>
    </section>
  </div>`;

  if (c.resp_prob != null) renderGauge(c);
  renderCompareChart(c, segAvg);
  renderSpendChart(c, segAvg);
  renderChannelChart(c, segAvg);
  renderLatentPosition(c, similar);
}

function renderGauge(c) {
  aiChart($('ai-gauge'), {
    series: [{
      type: 'gauge', startAngle: 200, endAngle: -20, min: 0, max: 100, radius: '95%', center: ['50%', '60%'],
      progress: { show: true, width: 14, itemStyle: { color: C.orange } },
      axisLine: { lineStyle: { width: 14, color: [[1, C.faint]] } },
      pointer: { show: false }, axisTick: { show: false }, splitLine: { show: false },
      axisLabel: { distance: -34, color: C.muted, fontSize: 9, formatter: v => (v % 50 === 0 ? `${v}%` : '') },
      title: { offsetCenter: [0, '32%'], color: C.muted, fontSize: 11 },
      detail: { offsetCenter: [0, '-2%'], fontSize: 30, fontWeight: 700, color: C.orange, formatter: v => `${v.toFixed(0)}%` },
      data: [{ value: c.resp_prob * 100, name: 'chance to respond' }],
    }],
  });
}

function renderCompareChart(c, segAvg) {
  const o = ai.overall;
  const CAP = 400;
  const idx = (v, avg) => (avg ? Math.min(CAP, (v / avg) * 100) : 100);
  const labels = AI_METRICS.map(m => m.label);
  aiChart($('ai-compare'), {
    textStyle: { fontFamily: FONT },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { color: C.muted, fontSize: 11 } },
    grid: { left: 12, right: 30, top: 26, bottom: 4, containLabel: true },
    tooltip: {
      ...TOOLTIP, trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(0,0,0,.03)' } },
      formatter: ps => {
        const m = AI_METRICS[ps[0].dataIndex];
        return `<b>${m.label}</b><br/>This customer: <b>${m.fmt(Number(c[m.key]) || 0)}</b><br/>` +
          `Segment average: ${m.fmt(segAvg[m.key])}<br/>All customers: ${m.fmt(o[m.key])}`;
      },
    },
    xAxis: { type: 'value', max: v => Math.min(CAP, Math.max(150, Math.ceil(v.max / 50) * 50)), splitLine: { lineStyle: { color: '#F3F3F3' } }, ...axisMuted },
    yAxis: { type: 'category', data: labels, inverse: true, ...axisMuted, axisLabel: { color: C.slate, fontSize: 11 } },
    series: [
      {
        name: 'This customer', type: 'bar', barGap: '15%', barMaxWidth: 11,
        itemStyle: { color: C.orange, borderRadius: [0, 3, 3, 0] },
        data: AI_METRICS.map(m => idx(Number(c[m.key]) || 0, o[m.key])),
        markLine: {
          symbol: 'none', silent: true, lineStyle: { type: 'dashed', color: C.grey },
          label: { formatter: 'Avg', position: 'start', color: C.muted, fontSize: 10 }, data: [{ xAxis: 100 }],
        },
      },
      {
        name: 'Segment average', type: 'bar', barMaxWidth: 11,
        itemStyle: { color: C.slate, borderRadius: [0, 3, 3, 0] },
        data: AI_METRICS.map(m => idx(segAvg[m.key], o[m.key])),
      },
    ],
  });
}

function renderSpendChart(c, segAvg) {
  const segTotal = sum(PRODUCTS, p => segAvg[p[1]]);
  const total = c.Total_Spending || 0;
  aiChart($('ai-spend'), {
    textStyle: { fontFamily: FONT },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { color: C.muted, fontSize: 11 } },
    grid: { left: 4, right: 30, top: 26, bottom: 4, containLabel: true },
    tooltip: {
      ...TOOLTIP, trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(0,0,0,.03)' } },
      formatter: ps => {
        const [, key] = PRODUCTS[ps[0].dataIndex];
        return `<b>${ps[0].name}</b><br/>This customer: <b>${fmtMoney(c[key])}</b> (${ps[0].value.toFixed(0)}%)<br/>Segment avg share: ${ps[1].value.toFixed(0)}%`;
      },
    },
    xAxis: { type: 'value', axisLabel: { color: C.muted, fontSize: 10, formatter: '{value}%' }, splitLine: { lineStyle: { color: '#F3F3F3' } }, axisTick: { show: false } },
    yAxis: { type: 'category', data: PRODUCTS.map(p => p[0]), inverse: true, ...axisMuted, axisLabel: { color: C.slate, fontSize: 11 } },
    series: [
      { name: 'Customer', type: 'bar', barGap: '15%', barMaxWidth: 10, itemStyle: { color: C.orange, borderRadius: [0, 3, 3, 0] },
        data: PRODUCTS.map(p => (total ? (c[p[1]] / total) * 100 : 0)) },
      { name: 'Segment', type: 'bar', barMaxWidth: 10, itemStyle: { color: C.slate, borderRadius: [0, 3, 3, 0] },
        data: PRODUCTS.map(p => (segTotal ? (segAvg[p[1]] / segTotal) * 100 : 0)) },
    ],
  });
}

function renderChannelChart(c, segAvg) {
  aiChart($('ai-channel'), {
    textStyle: { fontFamily: FONT },
    legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { color: C.muted, fontSize: 11 } },
    grid: { left: 4, right: 4, top: 30, bottom: 4, containLabel: true },
    tooltip: { ...TOOLTIP, trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: 'rgba(0,0,0,.03)' } }, valueFormatter: v => (Number.isInteger(v) ? v : v.toFixed(1)) },
    xAxis: { type: 'category', data: CHANNELS.map(ch => ch[0]), ...axisMuted, axisLabel: { color: C.slate, fontSize: 11 } },
    yAxis: { type: 'value', show: false },
    series: [
      { name: 'Customer', type: 'bar', barGap: '15%', barMaxWidth: 18, itemStyle: { color: C.orange, borderRadius: [3, 3, 0, 0] },
        label: { show: true, position: 'top', color: C.muted, fontSize: 10 }, data: CHANNELS.map(ch => c[ch[1]]) },
      { name: 'Segment avg', type: 'bar', barMaxWidth: 18, itemStyle: { color: C.slate, borderRadius: [3, 3, 0, 0] },
        label: { show: true, position: 'top', color: C.muted, fontSize: 10, formatter: p => p.value.toFixed(1) },
        data: CHANNELS.map(ch => segAvg[ch[1]]) },
    ],
  });
}

function renderLatentPosition(c, similar) {
  const series = state.segments.map(s => ({
    name: s.name, type: 'scatter', symbolSize: 4, silent: true,
    itemStyle: { color: state.segColor[s.name], opacity: 0.22 },
    data: state.customers.filter(x => x.cluster_name === s.name).map(x => [x.latent_x, x.latent_y]),
  }));
  series.push({
    name: 'Similar customers', type: 'scatter', symbolSize: 10, z: 5,
    itemStyle: { color: 'rgba(255,255,255,0)', borderColor: C.slate, borderWidth: 2 },
    data: similar.map(s => [s.c.latent_x, s.c.latent_y, s.c.ID]),
  });
  series.push({
    name: custId(c.ID), type: 'scatter', symbol: 'pin', symbolSize: 34, z: 6,
    itemStyle: { color: C.orange, borderColor: '#fff', borderWidth: 1 },
    label: { show: true, formatter: custId(c.ID), position: 'top', color: C.slate, fontWeight: 700, fontSize: 11 },
    data: [[c.latent_x, c.latent_y, c.ID]],
  });
  aiChart($('ai-latent'), {
    textStyle: { fontFamily: FONT },
    grid: { left: 4, right: 10, top: 10, bottom: 4, containLabel: true },
    tooltip: { ...TOOLTIP, trigger: 'item', formatter: p => (p.value[2] != null ? `${p.seriesName === 'Similar customers' ? 'Similar: ' : ''}<b>${custId(p.value[2])}</b>` : '') },
    xAxis: { type: 'value', scale: true, splitLine: { lineStyle: { color: '#F3F3F3' } }, ...axisMuted },
    yAxis: { type: 'value', scale: true, splitLine: { lineStyle: { color: '#F3F3F3' } }, ...axisMuted },
    series,
  });
}
