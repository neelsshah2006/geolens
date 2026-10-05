/* GeoLens front end. Plain JavaScript, no build step, no network calls beyond the local API. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const h = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (v, d = 0) => (v * 100).toFixed(d) + "%";
const num = (v, d = 2) => (v == null ? "—" : Number(v).toFixed(d));
const area = (m2) => (m2 == null ? "—" : m2 >= 1e6 ? (m2 / 1e6).toFixed(2) + " km²" : m2 >= 1e4 ? (m2 / 1e4).toFixed(1) + " ha" : Math.round(m2).toLocaleString() + " m²");

async function api(path, opts) {
  const r = await fetch("/api" + path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return r.json();
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 3200);
}
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch (_) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (_) {} },
};

const S = { scenes: [], overview: null, hits: [], selected: null, events: [], event: null, map: null, mapReady: false };

/* ================================================================= modes */
const MODES = ["search", "changes", "ingest", "evaluation"];
function setMode(m, sub) {
  if (!MODES.includes(m)) m = "search";
  $$(".modes button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === m)));
  MODES.forEach((x) => ($("#mode-" + x).hidden = x !== m));
  const hash = "#" + m + (sub ? "/" + sub : "");
  if (location.hash !== hash) history.replaceState(null, "", hash);
  if (m === "search" && S.map) setTimeout(() => S.map.resize(), 30);
  if (m === "changes") loadChanges(sub);
  if (m === "ingest") loadImagery();
  if (m === "evaluation") loadEvaluation();
}
$$(".modes button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
window.addEventListener("hashchange", () => { const [m, s] = location.hash.slice(1).split("/"); setMode(m, s); });

/* ================================================================= boot */
async function boot() {
  try {
    [S.overview, S.scenes] = await Promise.all([api("/overview"), api("/scenes")]);
  } catch (e) {
    $("#bar-status").textContent = "Cannot reach the GeoLens API";
    return;
  }
  const o = S.overview;
  $("#bar-status").textContent = `${o.scenes} scenes · ${o.tiles.toLocaleString()} tiles · ${o.sites} sites · ${o.area_km2.toLocaleString()} km²`;
  initSearch();
  refreshPending();
  const [m, s] = location.hash.slice(1).split("/");
  setMode(m || "search", s);
}
async function refreshPending() {
  try {
    const list = await api("/changes?status=supported&review=pending");
    $("#n-pending").textContent = list.length || "";
  } catch (_) {}
}

/* ================================================================= search */
const COVER = [["water", "Water", "#3b7ea1"], ["veg", "Vegetation", "#5b8a4a"], ["bare", "Bare soil", "#c49a7a"], ["urban", "Built-up", "#8d8f94"], ["dark", "Shadow / dark", "#39414a"]];
const EXAMPLES = ["reservoir water", "dense urban area", "cropland and bare soil", "forest without roads", "bright smooth surfaces"];

function initSearch() {
  $("#q-examples").innerHTML = EXAMPLES.map((e) => `<button type="button">${h(e)}</button>`).join("");
  $$("#q-examples button").forEach((b) => b.addEventListener("click", () => { $("#q").value = b.textContent; runSearch(); }));
  $("#f-scenes").insertAdjacentHTML("beforeend", S.scenes.map((s) =>
    `<label><input type="checkbox" value="${h(s.scene_id)}" checked> <span>${h(s.title)} <span class="hint">${h(s.acquisition_date)}</span></span></label>`).join(""));
  $("#f-q").addEventListener("input", (e) => ($("#f-q-out").textContent = Number(e.target.value).toFixed(2)));
  $("#q-form").addEventListener("submit", (e) => { e.preventDefault(); runSearch(); });
  $("#q-file").addEventListener("change", (e) => { if (e.target.files[0]) runSearch(e.target.files[0]); });
  initMap();
}

function filters() {
  const checked = $$("#f-scenes input:checked").map((i) => i.value);
  const f = { min_quality: Number($("#f-q").value) };
  if ($("#f-from").value) f.date_from = $("#f-from").value;
  if ($("#f-to").value) f.date_to = $("#f-to").value;
  if (checked.length && checked.length < S.scenes.length) f.scene_ids = checked;
  return f;
}

async function runSearch(file, byTile) {
  const text = $("#q").value.trim();
  const box = $("#results");
  if (!file && !byTile && !text) { $("#q").focus(); return; }
  $("#r-count").textContent = "Searching…"; $("#r-sub").textContent = "";
  try {
    let res;
    if (file) {
      const fd = new FormData();
      fd.append("file", file); fd.append("text", text); fd.append("limit", "36");
      res = await api("/search/upload", { method: "POST", body: fd });
      $("#q-file").value = "";
    } else {
      res = await post("/search", { text: text || null, point_id: byTile ?? null, filters: filters(), limit: 36 });
    }
    showResults(res, { byImage: !!file, byTile });
  } catch (e) {
    box.innerHTML = `<p class="empty">The search failed: ${h(e.message)}. Check that the API is running, then try again.</p>`;
    $("#r-count").textContent = "Search failed";
  }
}

function showResults(res, o = {}) {
  S.hits = res.results; S.selected = null;
  renderIntent(res, o);
  renderResults(res);
  drawHits();
  $("#tile-detail").hidden = true;
}

function renderIntent(res, o) {
  const el = $("#intent");
  const parts = [];
  if (o.byImage) parts.push("Matching the image you supplied");
  if (o.byTile) parts.push("Tiles that look like the selected tile");
  if (o.bySite) parts.push("Places that look like <b>" + h(o.bySite) + "</b>");
  const a = res.intent.attributes || [];
  if (a.length) parts.push("Understood as <b>" + a.map((w) => (w.negated ? "no " : "") + h(w.label)).join(", ") + "</b>");
  const notes = (res.notes || []).map((n) => `<div class="warn">${h(n)}</div>`).join("");
  if (!parts.length && !notes) { el.hidden = true; return; }
  el.hidden = false;
  el.innerHTML = parts.join(". ") + (parts.length ? "." : "") + notes;
}

function renderResults(res) {
  const box = $("#results");
  $("#r-count").textContent = res.total ? `${res.total} tiles` : "No matching tiles";
  $("#r-sub").textContent = res.total ? "best match first" : "";
  if (!res.total) {
    box.innerHTML = `<p class="empty">Nothing in the index matches. Loosen the filters, lower the minimum image quality, or describe the surface differently (water, forest, farmland, bare soil, built-up, road, cloud).</p>`;
    return;
  }
  box.innerHTML = res.results.map((r, i) => `
    <button class="frame" data-i="${i}" aria-pressed="false" title="${h(r.scene_title)}, ${h(r.date)}">
      <span class="img"><img loading="lazy" src="${h(r.image_url)}" alt="Tile ${h(r.tile_id)}"></span>
      <span class="rk">${r.rank}</span>
      <span class="cap"><span>${h(frameName(r.tile_id))}</span><span>${num(r.score)}</span></span>
      <span class="tags">${r.matched.length ? h(r.matched.slice(0, 2).join(", ")) : "&nbsp;"}</span>
    </button>`).join("");
  $$("#results .frame").forEach((b) => b.addEventListener("click", () => selectHit(Number(b.dataset.i), true)));
}
const frameName = (id) => { const m = id.match(/_(\d{3})(\d{3})_.*_r(\d+)_c(\d+)$/); return m ? `${m[1]}/${m[2]} r${m[3]}c${m[4]}` : id.slice(-10); };

function selectHit(i, fly) {
  S.selected = i;
  const r = S.hits[i];
  $$("#results .frame").forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.i) === i)));
  const card = $("#results .frame[aria-pressed=true]");
  if (card) card.scrollIntoView({ block: "nearest" });
  const mix = COVER.map(([k, , c]) => `<i style="width:${pct(r.attrs[k] || 0, 1)};background:${c}"></i>`).join("");
  $("#tile-detail").hidden = false;
  $("#tile-detail").innerHTML = `
    <button class="x" aria-label="Close tile details">×</button>
    <div class="pic"><img src="${h(r.image_url)}" alt="Tile ${h(r.tile_id)}"></div>
    <h3>${h(r.scene_title)}</h3>
    <div class="hint mono">${h(r.tile_id.split("_").slice(-2).join(" "))}</div>
    <div class="mix" title="Surface mix (rule-based, approximate)">${mix}</div>
    <div class="legend">${COVER.filter(([k]) => (r.attrs[k] || 0) >= 0.03).map(([k, l, c]) => `<span style="--c:${c}">${l} ${pct(r.attrs[k])}</span>`).join("")}</div>
    <dl>
      <dt>Acquired</dt><dd>${h(r.date)}</dd>
      <dt>Sensor</dt><dd>${h(r.sensor)}</dd>
      <dt>Centre</dt><dd>${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}</dd>
      <dt>Image quality</dt><dd>${num(r.quality)}</dd>
      <dt>Match</dt><dd>${num(r.score)} (look ${num(r.similarity)}${r.attribute_score ? ", terms " + num(r.attribute_score) : ""})</dd>
    </dl>
    <div class="acts"><button class="quiet" data-act="similar">Find similar tiles</button><button class="quiet" data-act="fit">Zoom to tile</button></div>`;
  $("#tile-detail .x").onclick = () => ($("#tile-detail").hidden = true);
  $("#tile-detail [data-act=similar]").onclick = () => { $("#q").value = ""; runSearch(null, r.point_id); };
  $("#tile-detail [data-act=fit]").onclick = () => fitPoly(r.polygon, 15);
  if (S.mapReady) {
    S.map.getSource("sel").setData({ type: "FeatureCollection", features: [{ type: "Feature", geometry: r.polygon, properties: {} }] });
    if (fly) fitPoly(r.polygon, 12.5, true);
  }
}

/* ---------- map ---------- */
function graticule(b) {
  const step = 0.1, f = [];
  for (let x = Math.floor(b[0] / step) * step; x <= b[2] + step; x += step) f.push([[x, b[1] - 1], [x, b[3] + 1]]);
  for (let y = Math.floor(b[1] / step) * step; y <= b[3] + step; y += step) f.push([[b[0] - 1, y], [b[2] + 1, y]]);
  return { type: "Feature", geometry: { type: "MultiLineString", coordinates: f }, properties: {} };
}
function unionBounds() {
  const bs = S.scenes.map((s) => s.bbox_wgs84).filter(Boolean);
  if (!bs.length) return [-1, -1, 1, 1];
  return [Math.min(...bs.map((b) => b[0])), Math.min(...bs.map((b) => b[1])), Math.max(...bs.map((b) => b[2])), Math.max(...bs.map((b) => b[3]))];
}
function initMap() {
  const ub = unionBounds();
  const map = new maplibregl.Map({
    container: "map", attributionControl: false, dragRotate: false, pitchWithRotate: false,
    style: { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#cdd4d7" } }] },
    bounds: ub, fitBoundsOptions: { padding: 40 },
  });
  S.map = map;
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-left");
  map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
  map.on("load", () => {
    map.addSource("grat", { type: "geojson", data: graticule(ub) });
    map.addLayer({ id: "grat", type: "line", source: "grat", paint: { "line-color": "#9aa6ac", "line-width": 0.6, "line-opacity": 0.7 } });
    S.scenes.forEach((s, i) => {
      const [w, so, e, n] = s.overlay_bounds;
      map.addSource("sc" + i, { type: "image", url: s.overlay_url, coordinates: [[w, n], [e, n], [e, so], [w, so]] });
      map.addLayer({ id: "sc" + i, type: "raster", source: "sc" + i, paint: { "raster-resampling": "linear", "raster-fade-duration": 0 } });
    });
    const foot = { type: "FeatureCollection", features: S.scenes.map((s) => { const [a, b, c, d] = s.bbox_wgs84; return { type: "Feature", properties: {}, geometry: { type: "Polygon", coordinates: [[[a, b], [c, b], [c, d], [a, d], [a, b]]] } }; }) };
    map.addSource("foot", { type: "geojson", data: foot });
    map.addLayer({ id: "foot", type: "line", source: "foot", paint: { "line-color": "#17222c", "line-width": 1, "line-dasharray": [3, 2] } });
    map.addSource("hits", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "hits-fill", type: "fill", source: "hits", paint: { "fill-color": "#e8531d", "fill-opacity": ["interpolate", ["linear"], ["get", "rank"], 1, 0.26, 36, 0.05] } });
    map.addLayer({ id: "hits-line", type: "line", source: "hits", paint: { "line-color": "#e8531d", "line-width": 1.6 } });
    map.addSource("sel", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "sel", type: "line", source: "sel", paint: { "line-color": "#ffffff", "line-width": 4 } });
    map.addLayer({ id: "sel2", type: "line", source: "sel", paint: { "line-color": "#1d4f7c", "line-width": 2 } });
    S.mapReady = true;
    drawHits();
  });
  map.on("mousemove", (e) => ($("#m-coord").textContent = `${e.lngLat.lat.toFixed(4)}°, ${e.lngLat.lng.toFixed(4)}°`));
  map.on("click", "hits-fill", (e) => { const r = e.features[0].properties.rank; selectHit(r - 1, false); });
  map.on("mouseenter", "hits-fill", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "hits-fill", () => (map.getCanvas().style.cursor = ""));
}
function drawHits() {
  if (!S.mapReady) return;
  const fc = { type: "FeatureCollection", features: S.hits.map((r) => ({ type: "Feature", geometry: r.polygon, properties: { rank: r.rank } })) };
  S.map.getSource("hits").setData(fc);
  S.map.getSource("sel").setData({ type: "FeatureCollection", features: [] });
  if (S.hits.length) {
    const xs = S.hits.flatMap((r) => r.polygon.coordinates[0].map((p) => p[0])), ys = S.hits.flatMap((r) => r.polygon.coordinates[0].map((p) => p[1]));
    S.map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: 60, duration: 500, maxZoom: 11 });
  }
}
function fitPoly(poly, maxZoom, ease) {
  const xs = poly.coordinates[0].map((p) => p[0]), ys = poly.coordinates[0].map((p) => p[1]);
  S.map.fitBounds([[Math.min(...xs), Math.min(...ys)], [Math.max(...xs), Math.max(...ys)]], { padding: 180, maxZoom, duration: ease ? 350 : 0 });
}

/* ================================================================= changes */
const TYPE_PLAIN = { construction: "construction", clearance: "clearance or removal", water_expansion: "water spreading", water_reduction: "water receding", road: "a new linear feature", other: "an unclassified change" };

async function loadChanges(selectId) {
  const status = $("#c-status [aria-pressed=true]").dataset.v;
  const review = $("#c-review [aria-pressed=true]").dataset.v;
  const q = new URLSearchParams();
  if (status) q.set("status", status);
  if (review) q.set("review", review);
  S.events = await api("/changes?" + q);
  const ul = $("#c-list");
  if (!S.events.length) {
    ul.innerHTML = `<li><p class="empty">No changes match these filters.</p></li>`;
  } else {
    ul.innerHTML = S.events.map((e) => `
      <li><button class="ev" data-id="${h(e.id)}">
        <span class="th">${e.thumb ? `<img src="${h(e.thumb)}" alt="">` : ""}</span>
        <span>
          <span class="t1"><span>${h(e.type_label || "Change")}</span><span class="conf">${num(e.confidence)}</span></span>
          <span class="t2">${h(e.site_name || e.site_id)}</span>
          <span class="t3"><span class="pill ${e.status}">${e.status}</span>${e.review ? `<span class="pill ${e.review.decision}">${e.review.decision}</span>` : ""}<span class="mono">${h(e.before_date)} → ${h(e.after_date || "")}</span></span>
        </span></button></li>`).join("");
    $$("#c-list .ev").forEach((b) => b.addEventListener("click", () => openChange(b.dataset.id)));
  }
  if (selectId) openChange(selectId);
  else if (S.event && S.events.some((e) => e.id === S.event)) markCurrent();
}
function markCurrent() { $$("#c-list .ev").forEach((b) => b.setAttribute("aria-current", String(b.dataset.id === S.event))); }
["#c-status", "#c-review"].forEach((sel) => $(sel).addEventListener("click", (e) => {
  const b = e.target.closest("button"); if (!b) return;
  $$(sel + " button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  loadChanges();
}));
document.addEventListener("keydown", (e) => {
  if ($("#mode-changes").hidden || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
  if (e.key !== "j" && e.key !== "k") return;
  const i = S.events.findIndex((x) => x.id === S.event);
  const n = S.events[Math.min(S.events.length - 1, Math.max(0, i + (e.key === "j" ? 1 : -1)))];
  if (n) openChange(n.id);
});

async function openChange(id) {
  S.event = id;
  markCurrent();
  history.replaceState(null, "", "#changes/" + id);
  const wb = $("#workbench");
  wb.innerHTML = `<p class="empty big">Loading evidence…</p>`;
  try {
    const ev = await api("/changes/" + id);
    const site = await api("/sites/" + ev.site_id);
    renderWorkbench(ev, site);
  } catch (e) {
    wb.innerHTML = `<p class="empty big">Could not load this change: ${h(e.message)}.</p>`;
  }
}

const FACTOR_TEXT = {
  evidence: ["Strength of change", "how far the pixels moved beyond normal noise"],
  quality: ["Usable imagery", "after masking cloud, shadow and haze"],
  registration: ["Alignment", "how well the two dates line up"],
  temporal: ["Persistence", "does it still show in later passes"],
  season: ["Season match", "penalised if only vegetation changed across seasons"],
  sensor: ["Same sensor", "mixed sensors reduce trust"],
  classification: ["Type rule fit", "how clearly the change type was identified"],
};

function renderWorkbench(ev, site) {
  const sup = ev.status === "supported";
  const thr = ev.params?.series?.conf_threshold ?? 0.55;
  const bd = ev.before, ad = ev.after;
  const synth = (o) => (o.synthetic ? " · synthetic" : "");
  const tr = ev.transitions || {};
  const regs = (ev.regions || []).slice().sort((a, b) => b.area_px - a.area_px);

  const why = narrative(ev, thr);
  const factors = Object.entries(ev.factors || {}).map(([k, v]) => `
    <span>${h(FACTOR_TEXT[k]?.[0] || k)}<small>${h(FACTOR_TEXT[k]?.[1] || "")}</small></span>
    <span class="bar"><i class="${v < 0.5 ? "low" : ""}" style="width:${pct(v)}"></i></span><span class="v">${num(v)}</span>`).join("");

  const regRows = regs.slice(0, 6).map((r) => {
    const d = bestDelta(r);
    return `<tr><td>${h(TYPE_PLAIN[r.type] || r.type)}</td><td class="n">${area(r.area_m2)}</td><td class="delta">${d}</td></tr>`;
  }).join("");

  const hist = (ev.reviews || []).slice().reverse().map((r) => `<li><span class="pill ${r.decision}">${r.decision}</span> ${h(r.analyst_id)} <time>${h(r.timestamp.slice(0, 16).replace("T", " "))}</time>${r.comment ? `<div>${h(r.comment)}</div>` : ""}</li>`).join("");

  $("#workbench").innerHTML = `
  <div class="wb">
    <div class="wb-main">
      <h1>${h(ev.type_label)} · ${h(ev.site_name)}</h1>
      <div class="sub mono">${h(ev.id)} · ${h(ev.window_start)} → ${h(ad.date)} · ${ev.gsd_m < 1 ? ev.gsd_m + " m" : Math.round(ev.gsd_m) + " m"} pixels</div>
      <div class="verdict">
        <span class="pill ${ev.status}">${sup ? "Supported" : "Suppressed"}</span>
        <span>Confidence <b class="mono">${num(ev.confidence)}</b>, reporting threshold <span class="mono">${num(thr)}</span></span>
        <span>${area(ev.area_m2)} changed${ev.area_frac != null ? ` (${pct(ev.area_frac, 1)} of the frame)` : ""}</span>
      </div>
      <div class="tl" id="tl"></div>
      <div class="table">
        <div class="curtain" id="curtain" tabindex="0" role="slider" aria-label="Before and after comparison" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50" style="--x:50%">
          <img class="before" src="${h(ev.images.before)}" alt="Before, ${h(bd.date)}">
          <img class="after" id="c-after" src="${h(ev.images.after)}" alt="After, ${h(ad.date)}">
          <span class="rule"></span><span class="grip"></span>
          <span class="lab l">Before ${h(bd.date)}${synth(bd)}</span><span class="lab r">After ${h(ad.date)}${synth(ad)}</span>
        </div>
        <div class="table-tools">
          <label><input type="checkbox" id="c-over"> Outline changed pixels</label>
          <span class="keys">Drag, or use ← → when focused. j / k moves between changes.</span>
        </div>
      </div>
      ${site.truth_url ? `<div class="truthrow"><img src="${h(site.truth_url)}" alt="Hand-drawn change mask"><p>Hand-drawn change mask from the LEVIR-CD dataset, for comparison. White is where annotators marked new buildings.</p></div>` : ""}
    </div>
    <aside class="wb-side">
      <section class="sec"><h2>Why GeoLens says this</h2><div class="why">${why}</div></section>
      <section class="sec"><h2>Confidence breakdown</h2><div class="factors">${factors}</div>
        <p class="hint" style="margin-top:8px">Each factor is 0 to 1. The tick marks 0.3: below it on quality, alignment, persistence or season, confidence is capped.</p></section>
      ${regRows ? `<section class="sec regions"><h2>Largest changed regions</h2><table class="grid"><tr><th>What</th><th class="n">Size</th><th>Surface change</th></tr>${regRows}</table>${regs.length > 6 ? `<p class="hint" style="margin-top:6px">${regs.length - 6} smaller regions not listed.</p>` : ""}</section>` : ""}
      <section class="sec review"><h2>Your decision</h2>
        <textarea id="r-comment" placeholder="Note what you checked or why you disagree (optional)"></textarea>
        <div class="row">
          <button class="confirm" data-d="confirmed">Confirm change</button>
          <button class="reject" data-d="rejected">Reject</button>
          <button class="flag" data-d="flagged">Flag for second look</button>
        </div>
        <div class="row" style="margin-top:6px">
          <button class="quiet" id="b-similar" ${site.georeferenced && site.gsd_m >= 10 ? "" : 'disabled title="This is a 0.5 m aerial chip. The index holds 30 m Landsat tiles, so similarity would not mean anything."'}>Find similar locations</button>
          <button class="quiet" id="b-export">Export case</button>
        </div>
        <div class="analyst" style="margin-top:8px"><label for="r-who">Reviewer</label><input id="r-who" type="text" value="${h(store.get("gl-analyst", "analyst"))}"></div>
        ${hist ? `<ul class="history">${hist}</ul>` : ""}
      </section>
      <section class="sec"><details class="prov" id="prov"><summary>Show processing record</summary><pre id="prov-body">Loading…</pre></details></section>
    </aside>
  </div>`;

  drawTimeline(ev, site);
  initCurtain(ev);
  $$(".review [data-d]").forEach((b) => b.addEventListener("click", () => submitReview(ev.id, b.dataset.d)));
  $("#b-export").addEventListener("click", () => { location.href = "/api/export/" + ev.id; });
  $("#b-similar").addEventListener("click", async () => {
    try {
      const res = await post(`/sites/${site.site_id}/similar`, { limit: 30 });
      setMode("search");
      $("#q").value = "";
      showResults(res, { bySite: site.name });
    } catch (e) { toast(e.message); }
  });
  $("#prov").addEventListener("toggle", async (e) => {
    if (!e.target.open) return;
    try { $("#prov-body").textContent = JSON.stringify(await api("/provenance/" + ev.id), null, 2); }
    catch (err) { $("#prov-body").textContent = err.message; }
  }, { once: true });
}

function bestDelta(r) {
  const keys = [["water", "water"], ["veg", "vegetation"], ["bare", "bare soil"], ["urban", "built-up"], ["dark", "dark surfaces"]];
  let best = null;
  for (const [k, l] of keys) {
    const d = (r.after?.[k] ?? 0) - (r.before?.[k] ?? 0);
    if (!best || Math.abs(d) > Math.abs(best.d)) best = { k, l, d };
  }
  if (!best || Math.abs(best.d) < 0.08) return `<span class="hint">no clear surface shift</span>`;
  return `${h(best.l)} ${pct(r.before[best.k])} → ${pct(r.after[best.k])}`;
}

function narrative(ev, thr) {
  const tr = ev.transitions || {};
  const reg = tr.registration || {};
  const out = [];
  const what = TYPE_PLAIN[ev.change_type] || "change";
  if (ev.status === "supported") {
    out.push(`Between <em>${h(ev.window_start)}</em> and <em>${h(ev.after.date)}</em> about ${area(ev.area_m2)} changed, and the most likely cause is <em>${h(what)}</em>. ${h(ev.after.date)} is the earliest usable pass that shows it, so that is the date reported.`);
  } else {
    out.push(`GeoLens found a difference between <em>${h(ev.window_start)}</em> and <em>${h(ev.after.date)}</em> but is <em>not reporting it</em> as a ${h(what)} event. Confidence is ${num(ev.confidence)} against a threshold of ${num(thr)}.`);
  }
  if (tr.persistence != null) out.push(`In later passes ${pct(tr.persistence)} of the changed pixels still look like the new state${tr.persistence >= 0.8 ? ", which is what a lasting change looks like" : ""}.`);
  else if (ev.status === "supported") out.push("There is no later pass to confirm it lasts, so persistence is assumed rather than measured.");
  if (reg.shift_dx != null && (Math.abs(reg.shift_dx) > 0.4 || Math.abs(reg.shift_dy) > 0.4)) out.push(`The later image sat ${num(Math.hypot(reg.shift_dx, reg.shift_dy), 1)} pixels off and was shifted back before comparing.`);
  if (reg.verified === false) out.push("The two dates differ too much to verify alignment from the imagery alone, so alignment is only partly trusted.");
  const rs = (ev.reasons || []).filter((r) => !/^latest transition|^confidence .* below threshold/.test(r));
  if (rs.length) out.push("Things that lowered trust: " + rs.map(h).join("; ") + ".");
  if (ev.site_note) out.push(`<span class="hint" style="font-family:var(--sans)">${h(ev.site_note)}</span>`);
  return out.map((p) => `<p>${p}</p>`).join("");
}

/* ---------- timeline ---------- */
function drawTimeline(ev, site) {
  const obs = site.observations_detail;
  const evs = site.events;
  const W = 720, padL = 28, padR = 28;
  const t = (d) => new Date(d + "T00:00:00Z").getTime();
  const t0 = t(obs[0].date), t1 = t(obs[obs.length - 1].date);
  const span = Math.max(t1 - t0, 1);
  const X = (d) => padL + ((t(d) - t0) / span) * (W - padL - padR);
  const levels = Math.max(1, evs.length);
  const axisY = 26 + levels * 11;
  const H = axisY + 54;
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Observation timeline">`;
  svg += `<line x1="${padL}" x2="${W - padR}" y1="${axisY}" y2="${axisY}" stroke="#4a5963" stroke-width="1"/>`;
  evs.slice().sort((a, b) => (a.after_date || "").localeCompare(b.after_date || "")).forEach((e, i) => {
    const y = axisY - 16 - i * 11, x0 = X(e.before_date), x1 = X(e.after_date || e.before_date);
    const sel = e.id === ev.id;
    const col = e.status === "supported" ? "#e8531d" : "#78858d";
    svg += `<g class="tl-ev" data-id="${h(e.id)}" style="cursor:pointer"><title>${h(e.type_label || "")} · ${h(e.status)} · confidence ${num(e.confidence)}</title>
      <rect x="${x0}" y="${y - 6}" width="${Math.max(x1 - x0, 4)}" height="12" fill="transparent"/>
      <path d="M${x0} ${y}H${x1}" stroke="${col}" stroke-width="${e.status === "supported" ? 4 : 2}" ${e.status === "supported" ? "" : 'stroke-dasharray="5 3"'} stroke-linecap="butt"/>
      <path d="M${x0} ${y - 4}v8M${x1} ${y - 4}v8" stroke="${col}" stroke-width="1.5"/>
      ${sel ? `<rect x="${x0 - 3}" y="${y - 7}" width="${Math.max(x1 - x0, 4) + 6}" height="14" fill="none" stroke="#17222c" stroke-width="1.2"/>` : ""}</g>`;
  });
  let lastX = -99, row = 0;
  obs.forEach((o) => {
    const x = X(o.date), q = o.quality || {}, excl = q.usable === false;
    row = x - lastX < 62 ? row + 1 : 0; lastX = x;
    const ly = axisY + 22 + (row % 2) * 14;
    svg += `<g><title>${h(o.date)} · ${h(o.sensor || "")}${o.synthetic ? " · synthetic epoch" : ""}${q.cloud_frac != null ? ` · cloud ${pct(q.cloud_frac)} · haze ${num(q.haze)}` : ""}${excl ? " · excluded: " + h(q.exclusion) : ""}\n${h(o.note || "")}</title>`;
    svg += excl
      ? `<circle cx="${x}" cy="${axisY}" r="6.5" fill="#edefee" stroke="#8a6d1e" stroke-width="1.8"/><path d="M${x - 4.5} ${axisY + 4.5}L${x + 4.5} ${axisY - 4.5}" stroke="#8a6d1e" stroke-width="1.8"/>`
      : `<circle cx="${x}" cy="${axisY}" r="6" fill="${o.synthetic ? "#edefee" : "#17222c"}" stroke="#17222c" stroke-width="1.8" ${o.synthetic ? 'stroke-dasharray="2.6 2"' : ""}/>${o.synthetic ? `<circle cx="${x}" cy="${axisY}" r="2" fill="#17222c"/>` : ""}`;
    svg += `<text x="${x}" y="${ly}" text-anchor="middle">${h(o.date)}</text></g>`;
  });
  svg += `</svg>`;
  const ex = obs.filter((o) => o.quality && o.quality.usable === false);
  const exNote = ex.length ? `<p class="excl-note"><b>Left out of the comparison:</b> ${ex.map((o) => `${h(o.date)} (${h(o.quality.exclusion)})`).join("; ")}.</p>` : "";
  $("#tl").innerHTML = svg + `<div class="tl-key"><span>● usable pass</span><span>◌ synthetic epoch (derived from a real image)</span><span style="color:#8a6d1e">⊘ excluded as unreliable</span><span style="color:#e8531d">▬ supported change</span><span>╌ suppressed candidate</span></div>` + exNote;
  $$("#tl .tl-ev").forEach((g) => g.addEventListener("click", () => openChange(g.dataset.id)));
}

/* ---------- curtain ---------- */
function initCurtain(ev) {
  const c = $("#curtain"), after = $("#c-after");
  let x = 50, drag = false;
  const set = (v) => { x = Math.max(0, Math.min(100, v)); c.style.setProperty("--x", x + "%"); c.setAttribute("aria-valuenow", Math.round(x)); };
  const at = (e) => { const r = c.getBoundingClientRect(); set(((e.clientX - r.left) / r.width) * 100); };
  c.addEventListener("pointerdown", (e) => { drag = true; c.setPointerCapture(e.pointerId); at(e); });
  c.addEventListener("pointermove", (e) => drag && at(e));
  c.addEventListener("pointerup", () => (drag = false));
  c.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { set(x - 4); e.preventDefault(); }
    if (e.key === "ArrowRight") { set(x + 4); e.preventDefault(); }
    if (e.key === "Home") set(0); if (e.key === "End") set(100);
  });
  $("#c-over").addEventListener("change", (e) => { after.src = e.target.checked ? ev.images.overlay : ev.images.after; });
}

async function submitReview(id, decision) {
  const who = $("#r-who").value.trim() || "analyst";
  store.set("gl-analyst", who);
  try {
    await post(`/changes/${id}/review`, { decision, comment: $("#r-comment").value.trim(), analyst_id: who });
    toast(decision === "confirmed" ? "Change confirmed" : decision === "rejected" ? "Change rejected" : "Flagged for a second look");
    refreshPending();
    await loadChanges();
    openChange(id);
  } catch (e) { toast("Could not save: " + e.message); }
}

/* ================================================================= imagery */
async function loadImagery() {
  const [scenes, runs] = await Promise.all([api("/scenes"), api("/runs?limit=12")]);
  $("#scene-table").innerHTML = `<tr><th>Scene</th><th>Acquired</th><th>Sensor</th><th class="n">Pixel</th><th class="n">Tiles</th><th>Footprint (W S E N)</th></tr>` +
    scenes.map((s) => `<tr><td>${h(s.title)}<div class="hint mono">${h(s.scene_id)}</div></td><td class="mono">${h(s.acquisition_date)}</td><td>${h(s.platform)}</td><td class="n">${s.resolution_m} m</td><td class="n">${s.qa?.tiles ?? ""}</td><td class="mono">${s.bbox_wgs84.map((v) => v.toFixed(2)).join("  ")}</td></tr>`).join("");
  $("#run-table").innerHTML = `<tr><th>Run</th><th>Kind</th><th>Subject</th><th>Result</th><th class="n">Seconds</th></tr>` +
    runs.map((r) => `<tr><td class="mono">${h(r.id)}</td><td>${h(r.kind)}</td><td>${h((r.scene_id || "").replace(/^LC08_L1TP_/, "").slice(0, 30))}</td><td class="${r.status === "done" ? "ok" : "no"}">${h(r.status)}${r.stats?.tiles ? ` · ${r.stats.tiles} tiles` : r.stats?.events != null ? ` · ${r.stats.events} events` : ""}</td><td class="n">${r.stats?.seconds ?? ""}</td></tr>`).join("");
}
(function ingestUi() {
  const drop = $("#drop"), file = $("#i-file");
  const show = () => { const f = file.files[0]; $("#drop-title").textContent = f ? f.name : "Choose a GeoTIFF"; $("#drop-sub").textContent = f ? (f.size / 1048576).toFixed(1) + " MB" : "or drag it onto this box"; };
  file.addEventListener("change", show);
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) { file.files = e.dataTransfer.files; show(); } });
  $("#ingest-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!file.files[0]) { toast("Choose a GeoTIFF first"); return; }
    const fd = new FormData();
    fd.append("file", file.files[0]); fd.append("date", $("#i-date").value); fd.append("sensor", $("#i-sensor").value);
    fd.append("replace", $("#i-replace").checked);
    $("#i-go").disabled = true;
    const job = $("#job"); job.hidden = false; job.className = "job";
    job.innerHTML = `<b>${h(file.files[0].name)}</b><div class="track"><i style="width:2%"></i></div><div class="msg hint">Uploading…</div>`;
    try {
      let j = await api("/ingest", { method: "POST", body: fd });
      while (j.status === "queued" || j.status === "running") {
        job.innerHTML = `<b>${h(j.file)}</b><div class="track"><i style="width:${pct(j.progress || 0.03)}"></i></div><div class="msg hint">${h(j.message)}</div>`;
        await new Promise((r) => setTimeout(r, 600));
        j = await api("/jobs/" + j.id);
      }
      if (j.status === "done") {
        job.innerHTML = `<b>${h(j.file)}</b><div class="msg">Added ${j.result.tiles} tiles in ${j.result.seconds} s. They are searchable now.</div>`;
        S.overview = await api("/overview"); S.scenes = await api("/scenes");
        $("#bar-status").textContent = `${S.overview.scenes} scenes · ${S.overview.tiles.toLocaleString()} tiles · ${S.overview.sites} sites · ${S.overview.area_km2.toLocaleString()} km²`;
        toast("Imagery added"); loadImagery();
      } else {
        job.classList.add("failed");
        job.innerHTML = `<b>${h(j.file)}</b><div class="msg">Not added: ${h(j.message)}</div>`;
      }
    } catch (err) {
      job.classList.add("failed"); job.innerHTML = `<div class="msg">Upload failed: ${h(err.message)}</div>`;
    }
    $("#i-go").disabled = false;
  });
})();

/* ================================================================= evaluation */
async function loadEvaluation() {
  const box = $("#eval-body");
  let m;
  try { m = await api("/evaluation"); } catch (e) { box.innerHTML = `<p class="empty">Could not load: ${h(e.message)}</p>`; return; }
  if (!m.available) { box.innerHTML = `<p class="empty">No evaluation has been run yet. Run <span class="mono">python scripts/build_demo.py</span> and reload.</p>`; return; }
  const r = m.retrieval, ss = m.scenario_summary;
  const pr = Object.entries(r.perturbation_recall).map(([k, v]) => `<tr><td>${h({ flip: "Mirrored", rot90: "Rotated 90°", brightness: "Brightness ±12%", noise: "Sensor noise", shift: "Shifted up to 6 px" }[k] || k)}</td><td class="n">${pct(v["recall@1"])}</td><td class="n">${pct(v["recall@5"])}</td><td class="n">${pct(v["recall@10"])}</td></tr>`).join("");
  box.innerHTML = `<div class="eval">
    <h2>How far to trust these results</h2>
    <p class="lede">Generated ${h(m.generated)} by ${h(m.pipeline)}. Every number below comes with the limit of what it can tell you.</p>

    <div class="block"><h2>Dating changes and rejecting false alarms</h2>
      <div class="figs"><div><b>${ss.passed} / ${ss.total}</b><span>scenarios correct</span></div></div>
      <table class="grid"><tr><th>Site</th><th>What it tests</th><th>Expected</th><th>Reported</th><th>Result</th></tr>
      ${m.scenarios.map((s) => `<tr><td>${h(s.name)}</td><td>${h(s.scenario.replace(/-/g, " "))}</td><td class="mono">${h(s.expected_date || "no change")}</td><td class="mono">${h(s.got_date || "no change")}${s.false_alarm_dates.length ? `<div class="no">false alarm ${h(s.false_alarm_dates.join(", "))}</div>` : ""}</td><td class="${s.pass ? "ok" : "no"}">${s.pass ? "correct" : "wrong"}</td></tr>`).join("")}</table>
      <p class="caveat">${h(m.scenario_caveat)}</p></div>

    <div class="block"><h2>Pixel accuracy on real change labels</h2>
      <div class="figs"><div><b>${num(m.levir_pixel_mean.f1)}</b><span>F1 (mean of 3)</span></div><div><b>${num(m.levir_pixel_mean.iou)}</b><span>IoU</span></div><div><b>${num(m.levir_pixel_mean.precision)}</b><span>precision</span></div><div><b>${num(m.levir_pixel_mean.recall)}</b><span>recall</span></div><div><b>${pct(m.levir_pixel_mean.fpr ?? 0, 1)}</b><span>unchanged pixels flagged</span></div></div>
      <table class="grid"><tr><th>LEVIR-CD pair</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">IoU</th><th class="n">False positive rate</th></tr>
      ${m.levir_pixel.map((p) => `<tr><td class="mono">${h(p.pair)}</td><td class="n">${num(p.precision)}</td><td class="n">${num(p.recall)}</td><td class="n">${num(p.f1)}</td><td class="n">${num(p.iou)}</td><td class="n">${pct(p.fpr ?? 0, 1)}</td></tr>`).join("")}</table>
      <p class="caveat">${h(m.levir_caveat)}</p></div>

    <div class="block"><h2>Search</h2>
      <div class="figs"><div><b>${r.search_ms_median} ms</b><span>median query, ${r.index_size} tiles</span></div><div><b>${num(r.label_coherence_p10 ?? r["label_coherence_p@10"])}</b><span>neighbour agreement (chance ${num(r.label_coherence_random_baseline)})</span></div></div>
      <table class="grid"><tr><th>Does a distorted copy find its original?</th><th class="n">Top 1</th><th class="n">Top 5</th><th class="n">Top 10</th></tr>${pr}</table>
      ${r.caveats.map((c) => `<p class="caveat">${h(c)}</p>`).join("")}</div>

    ${m.system ? `<div class="block"><h2>Size, speed and hardware</h2>
      <div class="figs"><div><b>${m.system.indexed_area_km2.toLocaleString()}</b><span>km² indexed</span></div><div><b>${m.system.tiles}</b><span>tiles, ${m.system.scenes} scenes</span></div><div><b>${m.system.ingest_seconds_per_scene} s</b><span>to add one scene</span></div><div><b>${m.system.ingest_ms_per_tile} ms</b><span>per tile, end to end</span></div></div>
      <table class="grid"><tr><th>Storage</th><th class="n">MB</th></tr>
      <tr><td>Processed imagery, tiles and evidence</td><td class="n">${m.system.storage_mb.processed_imagery}</td></tr>
      <tr><td>Vector index</td><td class="n">${m.system.storage_mb.vector_index}</td></tr>
      <tr><td>Raw input rasters</td><td class="n">${m.system.storage_mb.raw_inputs}</td></tr></table>
      <p class="hint" style="margin-top:8px">Measured on ${h(m.system.hardware.cpu_cores)} CPU cores, ${h(m.system.hardware.gpu)}, Python ${h(m.system.hardware.python)}. Adding a scene inserts its tiles only; nothing already indexed is recomputed.</p></div>` : ""}

    <div class="block"><h2>What this build does not do</h2>
      <p class="lede">The tile descriptor is hand-built from colour, texture and layout. It is not a trained remote-sensing model, and text search understands a fixed vocabulary of surface terms rather than free language. Pretrained encoders could not be downloaded in the environment that built this, so the embedder is a single replaceable module. Change epochs marked synthetic are derived from real images to test the logic; they are not observations.</p></div>
  </div>`;
}

boot();
