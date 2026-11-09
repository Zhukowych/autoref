/* Autoref paper map — Obsidian plugin (plain JS, no build step).
 *
 * Reads `.autoref/graph.json` (written by `autoref export obsidian`) and draws the papers as a
 * map: every top-level category is a translucent halo, its papers drift inside it, sub-categories
 * form denser clouds. Relations (citation edges labelled by the LLM) stay quiet until a paper is
 * hovered; clicking a relation opens the LLM's analysis of it.
 */
'use strict';
const { Plugin, ItemView, Notice, TFile, setIcon } = require('obsidian');

const VIEW = 'autoref-paper-map';
const DATA = '.autoref/graph.json';
const CAT_HUES = [199, 258, 36, 347, 164, 78, 300, 14];   // ice, violet, amber, coral, mint, lime, magenta, orange
const REL_HUES = [190, 48, 330, 140, 272, 18, 100, 220];
const NONE = '_none';
const BUILD = '0.1.2';
const dbg = (...a) => console.log('[autoref-graph ' + BUILD + ']', ...a);

/* ------------------------------------------------------------------ small helpers */
function h(tag, cls, text) {
  const el = document.createElement(tag);
  if (cls) el.className = cls;
  if (text != null) el.textContent = text;
  return el;
}
function seeded(id, salt) {            // deterministic pseudo random in [0,1) so the layout is stable
  let x = Math.imul(id * 2654435761 + salt * 40503, 2246822519) >>> 0;
  x ^= x >>> 15; x = Math.imul(x, 3266489917) >>> 0; x ^= x >>> 13;
  return (x >>> 0) / 4294967296;
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
function shortName(p) {
  const a = p.authors && p.authors.length ? p.authors[0].split(' ').pop() : 'Anon';
  return `${a}${p.n_authors > 1 ? ' et al.' : ''} ${p.year || ''}`.trim();
}

/* ------------------------------------------------------------------ model + layout */
/** Greedy circle packing: biggest first, each placed on a spiral at the first spot that fits. */
function packCircles(items, gap) {
  const placed = [];
  for (const c of [...items].sort((a, b) => b.R - a.R)) {
    for (let k = 0; ; k++) {
      const rho = k * 8, ang = k * 2.39996;
      const x = rho * Math.cos(ang), y = rho * Math.sin(ang);
      if (placed.every(o => Math.hypot(o.cx - x, o.cy - y) >= o.R + c.R + gap)) { c.cx = x; c.cy = y; break; }
    }
    placed.push(c);
  }
}

function buildModel(data) {
  const catById = new Map(data.categories.map((c, i) => [c.id, { ...c, order: i }]));
  const tops = data.categories.filter(c => !c.parent).map(c => c.id);
  const hasNone = data.papers.some(p => !p.primary || !catById.has(p.primary));
  if (hasNone) tops.push(NONE);
  const topOf = id => (catById.has(id) ? catById.get(id).top : NONE);

  const nodes = data.papers.map(p => {
    const primary = catById.has(p.primary) ? p.primary : null;
    const r = 3.2 + clamp(Math.sqrt(p.inbound || 0) * 1.5 + Math.log10(1 + (p.citations || 0)) * 0.45, 0, 7.5);
    return { id: p.id, p, r, top: primary ? topOf(primary) : NONE, sub: primary || NONE,
             x: 0, y: 0, vx: 0, vy: 0, dx: 0, dy: 0, phase: seeded(p.id, 3) * 6.283, vis: true, match: true };
  });
  const byId = new Map(nodes.map(n => [n.id, n]));

  const clusters = tops.map((id, i) => {
    const c = id === NONE ? { id, label: 'Uncategorized', description: '' } : catById.get(id);
    const mine = nodes.filter(n => n.top === id);
    return { id, label: c.label, description: c.description, idx: i, hue: id === NONE ? null : CAT_HUES[i % CAT_HUES.length],
             nodes: mine, n: mine.length, R: 0, cx: 0, cy: 0, subs: [] };
  }).filter(c => c.n > 0);

  // each sub-category is its own cloud; clouds are packed inside the halo, halos are packed on the canvas
  for (const c of clusters) {
    const subIds = [...new Set(c.nodes.map(n => n.sub))]
      .sort((a, b) => (catById.get(a)?.order ?? 1e9) - (catById.get(b)?.order ?? 1e9));
    c.subs = subIds.map(s => {
      const nodesOf = c.nodes.filter(n => n.sub === s);
      const label = s === NONE ? 'Uncategorized' : (s === c.id ? 'General' : catById.get(s).label);
      return { id: s, label, nodes: nodesOf, R: 10 * Math.sqrt(nodesOf.length) + 14, cx: 0, cy: 0, isTop: s === c.id };
    });
    packCircles(c.subs, 16);
    c.R = Math.max(...c.subs.map(s => Math.hypot(s.cx, s.cy) + s.R)) + 26;
    c.subs.forEach(s => { if (c.subs.length === 1) s.cx = s.cy = 0; });
  }
  packCircles(clusters, 70);
  for (const c of clusters) for (const s of c.subs) {
    s.tx = c.cx + s.cx; s.ty = c.cy + s.cy;
    for (const n of s.nodes) {
      n.tx = s.tx; n.ty = s.ty; n.cluster = c; n.subRef = s;
      const a = seeded(n.id, 1) * 6.283, d = Math.sqrt(seeded(n.id, 2)) * s.R * 0.6;
      n.x = n.tx + Math.cos(a) * d; n.y = n.ty + Math.sin(a) * d;
    }
  }

  const labels = data.relation_labels.map((l, i) => ({ ...l, idx: i }));
  const labelIdx = new Map(labels.map(l => [l.id, l.idx]));
  const links = [];
  for (const r of data.relations) {
    const a = byId.get(r.source), b = byId.get(r.target);
    if (!a || !b || a === b) continue;
    links.push({ r, a, b, li: labelIdx.has(r.label) ? labelIdx.get(r.label) : 0, vis: true });
  }
  const out = new Map(), inn = new Map();
  for (const l of links) {
    (out.get(l.a.id) || out.set(l.a.id, []).get(l.a.id)).push(l);
    (inn.get(l.b.id) || inn.set(l.b.id, []).get(l.b.id)).push(l);
  }
  return { data, catById, clusters, nodes, byId, links, labels, out, inn };
}

function simulate(model, iters, alpha0) {
  const { nodes, links, clusters } = model;
  let alpha = alpha0;
  for (let it = 0; it < iters; it++) {
    for (const n of nodes) {
      n.vx += (n.tx - n.x) * 0.06 * alpha; n.vy += (n.ty - n.y) * 0.06 * alpha;
    }
    for (const l of links) {                      // weak springs: relations bend the layout, never dominate it
      const dx = l.b.x - l.a.x, dy = l.b.y - l.a.y, d = Math.hypot(dx, dy) || 1;
      if (d > 120) {
        const f = (d - 120) * 0.0016 * alpha;
        l.a.vx += dx / d * f; l.a.vy += dy / d * f; l.b.vx -= dx / d * f; l.b.vy -= dy / d * f;
      }
    }
    const grid = new Map();
    for (const n of nodes) {
      const key = Math.floor(n.x / 40) * 100003 + Math.floor(n.y / 40);
      (grid.get(key) || grid.set(key, []).get(key)).push(n);
    }
    for (const n of nodes) {
      const cx = Math.floor(n.x / 40), cy = Math.floor(n.y / 40);
      for (let gx = cx - 1; gx <= cx + 1; gx++) for (let gy = cy - 1; gy <= cy + 1; gy++) {
        const cell = grid.get(gx * 100003 + gy);
        if (!cell) continue;
        for (const m of cell) {
          if (m.id <= n.id) continue;
          let dx = m.x - n.x, dy = m.y - n.y, d = Math.hypot(dx, dy);
          const min = (n.r + m.r) * 1.7 + 7;
          if (d < min) {
            if (d < 0.01) { dx = seeded(n.id, 9) - 0.5; dy = seeded(m.id, 9) - 0.5; d = Math.hypot(dx, dy) || 1; }
            const push = (min - d) * 0.5 * Math.min(1, alpha * 3 + 0.15);
            n.x -= dx / d * push; n.y -= dy / d * push; m.x += dx / d * push; m.y += dy / d * push;
          }
        }
      }
    }
    for (const n of nodes) {                                    // keep papers inside their cloud and halo
      const s = n.subRef, dx = n.x - s.tx, dy = n.y - s.ty, d = Math.hypot(dx, dy) || 1;
      if (d > s.R) { const f = (d - s.R) * 0.1; n.vx -= dx / d * f; n.vy -= dy / d * f; }
    }
    for (const n of nodes) {
      if (n.fixed) { n.vx = n.vy = 0; continue; }
      n.vx *= 0.8; n.vy *= 0.8; n.x += n.vx; n.y += n.vy;
    }
    alpha = Math.max(0.02, alpha * 0.985);
  }
}

/* ------------------------------------------------------------------ the view */
class PaperMapView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.model = null;
    this.cam = { x: 0, y: 0, k: 1 };
    this.camTo = null;
    this.hoverNode = this.hoverLink = this.selNode = this.selLink = null;
    this.hiddenCats = new Set();
    this.hiddenRels = new Set();
    this.minConf = 0;
    this.query = '';
    this.linesAtRest = true;
    this.motion = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.dirty = true;
    this.mtime = 0;
  }
  getViewType() { return VIEW; }
  getDisplayText() { return 'Paper map'; }
  getIcon() { return 'orbit'; }

  async onOpen() {
    const root = this.contentEl;
    root.empty();
    root.addClass('arm-root');
    this.canvas = root.appendChild(h('canvas', 'arm-canvas'));
    this.ctx = this.canvas.getContext('2d');
    this.tip = root.appendChild(h('div', 'arm-tip'));
    this.panel = root.appendChild(h('div', 'arm-panel'));
    this.inspector = root.appendChild(h('div', 'arm-inspector arm-hidden'));
    this.empty = root.appendChild(h('div', 'arm-empty arm-hidden'));

    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(root);
    this.bindPointer();
    this.registerEvent(this.app.workspace.on('css-change', () => { this.dirty = true; }));
    this.registerInterval(window.setInterval(() => this.checkFresh(), 15000));
    const loop = t => {
      try { this.frame(t); } catch (e) { if (!this._frameErr) { this._frameErr = true; console.error('[autoref-graph] draw failed', e); this.showEmpty('Drawing failed', String(e && e.stack || e)); } }
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
    dbg('view opened');
    try { await this.loadMap(true); }
    catch (e) { console.error('[autoref-graph] load failed', e); this.showEmpty('Loading the map failed', String(e && e.stack || e)); }
  }
  async onClose() {
    cancelAnimationFrame(this.raf);
    if (this.ro) this.ro.disconnect();
  }

  /* ---- data */
  async loadMap(fit) {
    const ad = this.app.vault.adapter;
    let raw;
    dbg('reading', DATA);
    try {
      raw = await ad.read(DATA);
      dbg('read', raw.length, 'bytes');
      const st = await ad.stat(DATA);
      this.mtime = st ? st.mtime : 0;
    } catch (e) {
      dbg('read failed', e);
      this.showEmpty('Cannot read ' + DATA, `${e && e.message || e}. Run \`autoref export obsidian -c <config>\` for THIS vault and reload.`);
      return;
    }
    let data;
    try { data = JSON.parse(raw); } catch (e) { this.showEmpty('Map data is unreadable', String(e)); return; }
    if (!data.papers || !data.papers.length) { this.showEmpty('No papers in the map yet', 'Run `autoref run` and export again.'); return; }
    this.empty.addClass('arm-hidden');
    const keepNode = this.selNode ? this.selNode.id : null;
    const keepLink = this.selLink ? this.selLink.r.id : null;
    this.model = buildModel(data);
    dbg('model', this.model.nodes.length, 'papers,', this.model.links.length, 'relations,', this.model.clusters.length, 'halos');
    simulate(this.model, 320, 1);
    this.selNode = keepNode != null ? this.model.byId.get(keepNode) || null : null;
    this.selLink = keepLink != null ? this.model.links.find(l => l.r.id === keepLink) || null : null;
    this.buildPanel();
    this.applyFilters();
    this.renderInspector();
    this.resize();
    if (fit) this.fit(false);
    dbg('ready; canvas', this.w, 'x', this.hgt, 'camera', JSON.stringify(this.cam));
    new Notice(`Paper map: ${this.model.nodes.length} papers, ${this.model.links.length} relations`);
  }
  async checkFresh() {
    if (!this.model || !this.containerEl.isShown()) return;
    try {
      const st = await this.app.vault.adapter.stat(DATA);
      if (st && st.mtime !== this.mtime) { await this.loadMap(false); new Notice('Paper map updated'); }
    } catch (e) { /* ignore */ }
  }
  showEmpty(title, body) {
    this.model = null;
    this.empty.empty();
    this.empty.appendChild(h('div', 'arm-empty-title', title));
    this.empty.appendChild(h('div', 'arm-empty-body', body));
    this.empty.removeClass('arm-hidden');
    this.panel.empty();
    this.dirty = true;
  }

  /* ---- theme */
  get dark() { return document.body.classList.contains('theme-dark'); }
  col(hue, a, role) {
    if (hue == null) return this.dark ? `hsla(220,10%,${role === 'soft' ? 62 : 72}%,${a})` : `hsla(220,10%,40%,${a})`;
    if (this.dark) return `hsla(${hue},${role === 'soft' ? 90 : 85}%,${role === 'soft' ? 60 : 70}%,${a})`;
    return `hsla(${hue},${role === 'soft' ? 75 : 70}%,${role === 'soft' ? 55 : 42}%,${a})`;
  }
  relCol(i, a) { return this.col(REL_HUES[i % REL_HUES.length], a); }
  catHue(n) { return n.cluster ? n.cluster.hue : null; }

  /* ---- layout of the canvas */
  resize() {
    const r = this.contentEl.getBoundingClientRect();
    this.w = Math.max(200, r.width); this.hgt = Math.max(200, r.height);
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(this.w * this.dpr); this.canvas.height = Math.round(this.hgt * this.dpr);
    this.canvas.style.width = this.w + 'px'; this.canvas.style.height = this.hgt + 'px';
    this.dirty = true;
  }
  visibleNodes() { return this.model.nodes.filter(n => n.vis); }
  fit(animate) {
    if (!this.model) return;
    const ns = this.visibleNodes();
    if (!ns.length) return;
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const c of this.model.clusters) {
      if (!c.nodes.some(n => n.vis)) continue;
      x0 = Math.min(x0, c.cx - c.R); x1 = Math.max(x1, c.cx + c.R); y0 = Math.min(y0, c.cy - c.R); y1 = Math.max(y1, c.cy + c.R);
    }
    const pad = 70, k = clamp(Math.min((this.w - pad * 2) / (x1 - x0), (this.hgt - pad * 2) / (y1 - y0)), 0.15, 2.2);
    const target = { k, x: this.w / 2 - (x0 + x1) / 2 * k, y: this.hgt / 2 - (y0 + y1) / 2 * k };
    if (animate) this.camTo = target; else { this.cam = target; this.dirty = true; }
  }
  focusOn(n) {
    const k = Math.max(this.cam.k, 1.6);
    this.camTo = { k, x: this.w / 2 - n.x * k, y: this.hgt / 2 - n.y * k };
  }
  toWorld(px, py) { return [(px - this.cam.x) / this.cam.k, (py - this.cam.y) / this.cam.k]; }

  /* ---- filters */
  catHidden(id) {
    for (let c = this.model.catById.get(id); c; c = c.parent ? this.model.catById.get(c.parent) : null)
      if (this.hiddenCats.has(c.id)) return true;
    return this.hiddenCats.has(id);
  }
  applyFilters() {
    if (!this.model) return;
    const q = this.query.trim().toLowerCase();
    for (const n of this.model.nodes) {
      n.vis = !(this.catHidden(n.sub) || (n.top === NONE && this.hiddenCats.has(NONE)));
      const p = n.p;
      n.match = !q || p.title.toLowerCase().includes(q) || (p.authors || []).join(' ').toLowerCase().includes(q) || String(p.year || '').includes(q);
    }
    for (const l of this.model.links) {
      l.vis = l.a.vis && l.b.vis && !this.hiddenRels.has(l.r.label) && (l.r.confidence || 0) >= this.minConf;
    }
    this.dirty = true;
  }

  /* ---- rendering */
  frame(t) {
    if (!this.model || !this.containerEl.isShown()) return;
    if (this.camTo) {
      const c = this.cam, g = this.camTo, e = 0.16;
      c.x += (g.x - c.x) * e; c.y += (g.y - c.y) * e; c.k += (g.k - c.k) * e;
      if (Math.abs(g.x - c.x) + Math.abs(g.y - c.y) < 0.4 && Math.abs(g.k - c.k) < 0.002) { this.cam = g; this.camTo = null; }
      this.dirty = true;
    }
    if (this.dragNode || this.reheat > 0) {
      simulate(this.model, 1, 0.12);
      if (this.reheat > 0) this.reheat--;
      this.dirty = true;
    }
    if (!this.dirty && !this.motion) return;
    this.draw(t);
    this.dirty = false;
  }
  draw(t) {
    const { ctx, model, cam } = this;
    const dark = this.dark;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.w, this.hgt);
    const drift = this.motion ? 1 : 0;
    for (const n of model.nodes) {
      n.dx = n.x + Math.sin(t * 0.00055 + n.phase) * 2.4 * drift;
      n.dy = n.y + Math.cos(t * 0.00047 + n.phase * 1.3) * 2.4 * drift;
    }
    ctx.setTransform(this.dpr * cam.k, 0, 0, this.dpr * cam.k, this.dpr * cam.x, this.dpr * cam.y);
    const k = cam.k;
    const glow = dark ? 'lighter' : 'source-over';

    // focus = what the user is looking at; everything else steps back
    const fl = this.selLink, fn = fl ? null : (this.hoverNode || this.selNode);
    const near = new Set(), nearLinks = new Set();
    if (fl) { near.add(fl.a); near.add(fl.b); nearLinks.add(fl); }
    else if (fn) {
      near.add(fn);
      for (const l of [...(model.out.get(fn.id) || []), ...(model.inn.get(fn.id) || [])]) if (l.vis) { near.add(l.a); near.add(l.b); nearLinks.add(l); }
    }
    const focus = !!(fl || fn);

    // halos
    ctx.globalCompositeOperation = glow;
    const stats = [];
    for (const c of model.clusters) {
      const vs = c.nodes.filter(n => n.vis);
      if (!vs.length) continue;
      let mx = 0, my = 0;
      for (const n of vs) { mx += n.dx; my += n.dy; }
      mx /= vs.length; my /= vs.length;
      let rad = 0;
      for (const n of vs) rad = Math.max(rad, Math.hypot(n.dx - mx, n.dy - my) + n.r);
      rad = rad * 1.18 + 38;
      stats.push({ c, mx, my, rad, count: vs.length });
      const g = ctx.createRadialGradient(mx, my, 0, mx, my, rad);
      const a0 = dark ? 0.2 : 0.22;
      g.addColorStop(0, this.col(c.hue, a0 * (focus ? 0.6 : 1), 'soft'));
      g.addColorStop(0.6, this.col(c.hue, a0 * 0.42 * (focus ? 0.6 : 1), 'soft'));
      g.addColorStop(1, this.col(c.hue, 0, 'soft'));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(mx, my, rad, 0, 6.283); ctx.fill();
      for (const s of c.subs) {                         // denser sub-clouds
        const sv = s.nodes.filter(n => n.vis);
        if (c.subs.length < 2 || sv.length < 2) continue;
        let sx = 0, sy = 0; for (const n of sv) { sx += n.dx; sy += n.dy; }
        sx /= sv.length; sy /= sv.length;
        let sr = 0; for (const n of sv) sr = Math.max(sr, Math.hypot(n.dx - sx, n.dy - sy) + n.r);
        sr = sr * 1.25 + 22;
        const sg = ctx.createRadialGradient(sx, sy, 0, sx, sy, sr);
        const hue = c.hue == null ? null : (c.hue + 16) % 360;
        sg.addColorStop(0, this.col(hue, (dark ? 0.16 : 0.12) * (focus ? 0.5 : 1), 'soft'));
        sg.addColorStop(1, this.col(hue, 0, 'soft'));
        ctx.fillStyle = sg; ctx.beginPath(); ctx.arc(sx, sy, sr, 0, 6.283); ctx.fill();
        s._lab = { x: sx, y: sy - sr * 0.82, count: sv.length };
      }
      ctx.strokeStyle = this.col(c.hue, dark ? 0.13 : 0.2, 'soft'); ctx.lineWidth = 1 / k;
      ctx.beginPath(); ctx.arc(mx, my, rad * 0.98, 0, 6.283); ctx.stroke();
    }
    ctx.globalCompositeOperation = 'source-over';

    // lines at rest: one quiet batch
    const quad = l => {
      const ax = l.a.dx, ay = l.a.dy, bx = l.b.dx, by = l.b.dy;
      const mx = (ax + bx) / 2, my = (ay + by) / 2, nx = -(by - ay), ny = bx - ax;
      return [ax, ay, mx + nx * 0.14, my + ny * 0.14, bx, by];
    };
    if (this.linesAtRest) {
      ctx.beginPath();
      for (const l of model.links) {
        if (!l.vis || nearLinks.has(l)) continue;
        const [ax, ay, cx, cy, bx, by] = quad(l);
        ctx.moveTo(ax, ay); ctx.quadraticCurveTo(cx, cy, bx, by);
      }
      ctx.strokeStyle = dark ? `rgba(190,205,240,${focus ? 0.025 : 0.07})` : `rgba(50,60,95,${focus ? 0.03 : 0.1})`;
      ctx.lineWidth = 0.9 / k; ctx.stroke();
    }
    // highlighted lines in their relation colour, with a slow flow from citing to cited paper
    ctx.globalCompositeOperation = glow;
    for (const l of nearLinks) {
      const [ax, ay, cx, cy, bx, by] = quad(l);
      const sel = l === this.selLink, hov = l === this.hoverLink;
      ctx.strokeStyle = this.relCol(l.li, sel ? 0.95 : hov ? 0.9 : 0.62);
      ctx.lineWidth = (sel ? 2.6 : hov ? 2.2 : 1.4) / k;
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.quadraticCurveTo(cx, cy, bx, by); ctx.stroke();
      if (this.motion) {
        const u = ((t * 0.00035) + l.r.id * 0.37) % 1, v = 1 - u;
        const px = v * v * ax + 2 * v * u * cx + u * u * bx, py = v * v * ay + 2 * v * u * cy + u * u * by;
        ctx.fillStyle = this.relCol(l.li, 0.95);
        ctx.beginPath(); ctx.arc(px, py, 2.4 / Math.sqrt(k), 0, 6.283); ctx.fill();
      }
    }
    ctx.globalCompositeOperation = 'source-over';

    // papers
    const sorted = model.nodes.filter(n => n.vis).sort((a, b) => a.r - b.r);
    for (const n of sorted) {
      const lit = !focus || near.has(n);
      const dim = (!n.match ? 0.16 : lit ? 1 : 0.22);
      const hue = this.catHue(n);
      ctx.globalCompositeOperation = glow;
      ctx.fillStyle = this.col(hue, 0.16 * dim, 'soft');
      ctx.beginPath(); ctx.arc(n.dx, n.dy, n.r * 2.7, 0, 6.283); ctx.fill();
      ctx.globalCompositeOperation = 'source-over';
      ctx.fillStyle = this.col(hue, 0.95 * dim);
      ctx.beginPath(); ctx.arc(n.dx, n.dy, n.r, 0, 6.283); ctx.fill();
      if (n.p.new) {
        ctx.strokeStyle = this.col(hue, 0.7 * dim); ctx.lineWidth = 1 / k;
        ctx.beginPath(); ctx.arc(n.dx, n.dy, n.r + 3.5, 0, 6.283); ctx.stroke();
      }
      if (n === this.selNode || n === this.hoverNode) {
        ctx.strokeStyle = dark ? 'rgba(255,255,255,0.9)' : 'rgba(20,20,30,0.85)'; ctx.lineWidth = 1.6 / k;
        ctx.beginPath(); ctx.arc(n.dx, n.dy, n.r + 3, 0, 6.283); ctx.stroke();
      }
    }

    // text in screen space so it stays readable at any zoom
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const font = getComputedStyle(this.contentEl).getPropertyValue('--font-interface').trim() || 'sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    for (const s of stats) {
      const x = s.mx * k + cam.x, y = (s.my - s.rad) * k + cam.y;
      const size = clamp(12 + Math.sqrt(s.rad) * 0.55 * Math.min(k, 1.4), 13, 26);
      ctx.font = `600 ${size}px ${font}`;
      ctx.fillStyle = this.col(s.c.hue, focus ? 0.4 : 0.92);
      ctx.fillText(s.c.label, x, y - 2);
      ctx.font = `400 ${Math.max(11, size * 0.6)}px ${font}`;
      ctx.fillStyle = dark ? `rgba(200,210,235,${focus ? 0.25 : 0.55})` : `rgba(60,70,100,${focus ? 0.3 : 0.65})`;
      ctx.fillText(`${s.count} papers`, x, y + size * 0.78);
      if (k > 0.45) for (const sub of s.c.subs) {
        if (!sub._lab || s.c.subs.length < 2) continue;
        ctx.font = `500 ${clamp(10.5 + k, 11, 14)}px ${font}`;
        ctx.fillStyle = this.col(s.c.hue, clamp((k - 0.45) * 1.6, 0, 0.8) * (focus ? 0.4 : 1));
        ctx.fillText(sub.label, sub._lab.x * k + cam.x, sub._lab.y * k + cam.y);
      }
    }
    ctx.font = `500 12px ${font}`;
    for (const n of sorted) {
      const show = n === this.hoverNode || n === this.selNode || (focus && near.has(n) && (near.size <= 9 || k > 1.3)) || (!focus && k > 2.6) || (this.query && n.match && n.vis);
      if (!show) continue;
      const x = n.dx * k + cam.x, y = n.dy * k + cam.y + (n.r + 11) * k * 0.9 + 2;
      const txt = shortName(n.p);
      ctx.lineWidth = 3; ctx.strokeStyle = dark ? 'rgba(8,10,18,0.75)' : 'rgba(255,255,255,0.8)';
      ctx.strokeText(txt, x, y);
      ctx.fillStyle = dark ? 'rgba(235,240,255,0.95)' : 'rgba(20,24,40,0.95)';
      ctx.fillText(txt, x, y);
    }
    // relation name next to the hovered / selected line
    const ll = this.selLink || this.hoverLink;
    if (ll && ll.vis) {
      const [ax, ay, cx, cy, bx, by] = quad(ll);
      const mx = (ax + 2 * cx + bx) / 4 * k + cam.x, my = (ay + 2 * cy + by) / 4 * k + cam.y;
      const lab = this.model.labels[ll.li];
      const txt = lab ? lab.label : ll.r.label;
      ctx.font = `600 12px ${font}`;
      const w = ctx.measureText(txt).width + 18;
      ctx.fillStyle = dark ? 'rgba(10,12,22,0.82)' : 'rgba(255,255,255,0.9)';
      ctx.beginPath(); ctx.roundRect(mx - w / 2, my - 11, w, 22, 11); ctx.fill();
      ctx.strokeStyle = this.relCol(ll.li, 0.9); ctx.lineWidth = 1; ctx.stroke();
      ctx.fillStyle = this.relCol(ll.li, 1);
      ctx.fillText(txt, mx, my + 0.5);
    }
  }

  /* ---- pointer */
  bindPointer() {
    const cv = this.canvas;
    const pos = e => { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    cv.addEventListener('wheel', e => {
      e.preventDefault();
      const [px, py] = pos(e);
      const k = clamp(this.cam.k * Math.exp(-e.deltaY * 0.0014), 0.08, 6);
      const [wx, wy] = this.toWorld(px, py);
      this.camTo = null;
      this.cam = { k, x: px - wx * k, y: py - wy * k };
      this.dirty = true;
    }, { passive: false });
    let down = null;
    cv.addEventListener('pointerdown', e => {
      if (!this.model) return;
      cv.setPointerCapture(e.pointerId);
      const [px, py] = pos(e);
      down = { px, py, moved: false, node: this.pickNode(px, py), cx: this.cam.x, cy: this.cam.y };
      if (down.node) { this.dragNode = down.node; down.node.fixed = true; }
    });
    cv.addEventListener('pointermove', e => {
      if (!this.model) return;
      const [px, py] = pos(e);
      if (down) {
        if (Math.hypot(px - down.px, py - down.py) > 3) down.moved = true;
        if (down.moved) {
          if (this.dragNode) { const [wx, wy] = this.toWorld(px, py); this.dragNode.x = wx; this.dragNode.y = wy; }
          else { this.camTo = null; this.cam.x = down.cx + (px - down.px); this.cam.y = down.cy + (py - down.py); }
          this.dirty = true;
        }
        return;
      }
      const n = this.pickNode(px, py);
      const l = n ? null : this.pickLink(px, py);
      if (n !== this.hoverNode || l !== this.hoverLink) {
        this.hoverNode = n; this.hoverLink = l; this.dirty = true;
        cv.style.cursor = n || l ? 'pointer' : 'grab';
        this.showTip(n, px, py);
      } else if (n) this.moveTip(px, py);
    });
    cv.addEventListener('pointerup', e => {
      if (!down) return;
      const d = down; down = null;
      if (this.dragNode) { this.dragNode.fixed = false; this.dragNode = null; this.reheat = 40; }
      if (d.moved) return;
      const [px, py] = pos(e);
      const n = this.pickNode(px, py), l = n ? null : this.pickLink(px, py);
      if (n) this.select(n, null); else if (l) this.select(null, l); else this.select(null, null);
    });
    cv.addEventListener('dblclick', e => {
      const [px, py] = pos(e);
      const n = this.pickNode(px, py);
      if (n) this.openNote(n);
    });
    cv.addEventListener('pointerleave', () => { if (!down) { this.hoverNode = this.hoverLink = null; this.hideTip(); this.dirty = true; } });
    this.registerDomEvent(window, 'keydown', e => { if (e.key === 'Escape' && this.containerEl.isShown() && !(e.target instanceof HTMLInputElement)) this.select(null, null); });
    cv.style.cursor = 'grab';
  }
  pickNode(px, py) {
    if (!this.model) return null;
    const [wx, wy] = this.toWorld(px, py), pad = 5 / this.cam.k;
    let best = null, bd = 1e9;
    for (const n of this.model.nodes) {
      if (!n.vis) continue;
      const d = Math.hypot(n.dx - wx, n.dy - wy);
      if (d <= n.r + pad && d < bd) { best = n; bd = d; }
    }
    return best;
  }
  pickLink(px, py) {
    const [wx, wy] = this.toWorld(px, py), tol = 7 / this.cam.k;
    const focus = this.selLink ? null : this.selNode;
    let best = null, bd = tol;
    for (const l of this.model.links) {
      if (!l.vis) continue;
      if (focus && l.a !== focus && l.b !== focus) continue;
      if (!this.linesAtRest && !focus && l !== this.selLink) continue;
      const ax = l.a.dx, ay = l.a.dy, bx = l.b.dx, by = l.b.dy;
      if (Math.min(ax, bx) - tol > wx || Math.max(ax, bx) + tol < wx || Math.min(ay, by) - tol > wy || Math.max(ay, by) + tol < wy) {
        // control point can bulge outside the chord's box a little; allow for it
        const bulge = Math.hypot(bx - ax, by - ay) * 0.07;
        if (Math.min(ax, bx) - tol - bulge > wx || Math.max(ax, bx) + tol + bulge < wx ||
            Math.min(ay, by) - tol - bulge > wy || Math.max(ay, by) + tol + bulge < wy) continue;
      }
      const mx = (ax + bx) / 2, my = (ay + by) / 2, cx = mx - (by - ay) * 0.14, cy = my + (bx - ax) * 0.14;
      let px0 = ax, py0 = ay;
      for (let i = 1; i <= 12; i++) {
        const u = i / 12, v = 1 - u;
        const x = v * v * ax + 2 * v * u * cx + u * u * bx, y = v * v * ay + 2 * v * u * cy + u * u * by;
        const d = segDist(wx, wy, px0, py0, x, y);
        if (d < bd) { bd = d; best = l; }
        px0 = x; py0 = y;
      }
    }
    return best;
  }
  showTip(n, px, py) {
    if (!n) { this.hideTip(); return; }
    this.tip.empty();
    this.tip.appendChild(h('div', 'arm-tip-title', n.p.title));
    this.tip.appendChild(h('div', 'arm-tip-sub', `${shortName(n.p)}${n.p.venue ? ' · ' + n.p.venue : ''}`));
    this.tip.addClass('arm-on');
    this.moveTip(px, py);
  }
  moveTip(px, py) {
    const w = this.tip.offsetWidth, hh = this.tip.offsetHeight;
    this.tip.style.left = clamp(px + 16, 8, this.w - w - 8) + 'px';
    this.tip.style.top = clamp(py + 16, 8, this.hgt - hh - 8) + 'px';
  }
  hideTip() { this.tip.removeClass('arm-on'); }

  select(node, link) {
    this.selNode = node; this.selLink = link;
    if (node) this.focusOn(node);
    this.renderInspector();
    this.dirty = true;
  }
  async openNote(n) {
    const f = this.app.vault.getAbstractFileByPath(n.p.note);
    if (f instanceof TFile) await this.app.workspace.getLeaf('tab').openFile(f);
    else new Notice(`Note not found: ${n.p.note}`);
  }

  /* ---- left panel */
  buildPanel() {
    const m = this.model, P = this.panel;
    P.empty();
    const head = P.appendChild(h('div', 'arm-panel-head'));
    head.appendChild(h('div', 'arm-title', m.data.topic.replace(/_/g, ' ')));
    const toggle = head.appendChild(h('button', 'arm-icon-btn'));
    setIcon(toggle, 'chevrons-up-down');
    toggle.setAttribute('aria-label', 'Collapse');
    toggle.onclick = () => P.toggleClass('arm-collapsed', !P.hasClass('arm-collapsed'));
    P.appendChild(h('div', 'arm-count', `${m.nodes.length} papers · ${m.links.length} relations · plugin ${BUILD}`));
    const body = P.appendChild(h('div', 'arm-panel-body'));

    const search = body.appendChild(h('input', 'arm-search'));
    search.type = 'search'; search.placeholder = 'Search title, author, year'; search.value = this.query;
    search.oninput = () => { this.query = search.value; this.applyFilters(); };

    body.appendChild(h('div', 'arm-section', 'Categories'));
    const list = body.appendChild(h('div', 'arm-list'));
    const countOf = new Map();
    for (const n of m.nodes) {
      let id = n.sub;
      while (id && id !== NONE) { countOf.set(id, (countOf.get(id) || 0) + 1); const c = m.catById.get(id); id = c ? c.parent : null; }
      if (n.sub === NONE) countOf.set(NONE, (countOf.get(NONE) || 0) + 1);
    }
    const rows = m.data.categories.filter(c => countOf.has(c.id)).map(c => ({ id: c.id, label: c.label, depth: c.depth, hue: m.clusters.find(x => x.id === c.top)?.hue }));
    if (countOf.has(NONE)) rows.push({ id: NONE, label: 'Uncategorized', depth: 0, hue: null });
    for (const r of rows) {
      const row = list.appendChild(h('div', 'arm-row'));
      row.style.paddingLeft = 6 + r.depth * 14 + 'px';
      const dot = row.appendChild(h('span', 'arm-dot'));
      dot.style.background = this.col(r.hue, 1); dot.style.color = this.col(r.hue, 1);
      row.appendChild(h('span', 'arm-row-label', r.label));
      row.appendChild(h('span', 'arm-row-count', String(countOf.get(r.id))));
      row.toggleClass('arm-off', this.hiddenCats.has(r.id));
      row.setAttribute('title', 'Click to hide or show. Shift-click to show only this one.');
      row.onclick = e => {
        if (e.shiftKey) { this.hiddenCats = new Set(rows.filter(x => x.id !== r.id && x.depth === 0).map(x => x.id)); }
        else if (this.hiddenCats.has(r.id)) this.hiddenCats.delete(r.id); else this.hiddenCats.add(r.id);
        this.buildPanel(); this.applyFilters(); this.fit(true);
      };
    }

    body.appendChild(h('div', 'arm-section', 'Relations'));
    const rl = body.appendChild(h('div', 'arm-list'));
    const used = new Map();
    for (const l of m.links) used.set(l.r.label, (used.get(l.r.label) || 0) + 1);
    for (const lab of m.labels) {
      if (!used.has(lab.id)) continue;
      const row = rl.appendChild(h('div', 'arm-row'));
      const dot = row.appendChild(h('span', 'arm-dash'));
      dot.style.background = this.relCol(lab.idx, 1);
      row.appendChild(h('span', 'arm-row-label', lab.label));
      row.appendChild(h('span', 'arm-row-count', String(used.get(lab.id))));
      row.toggleClass('arm-off', this.hiddenRels.has(lab.id));
      row.setAttribute('title', lab.description);
      row.onclick = () => { if (this.hiddenRels.has(lab.id)) this.hiddenRels.delete(lab.id); else this.hiddenRels.add(lab.id); this.buildPanel(); this.applyFilters(); };
    }
    const conf = body.appendChild(h('label', 'arm-slider'));
    const cv = conf.appendChild(h('span', null, `Minimum confidence ${this.minConf.toFixed(2)}`));
    const rng = conf.appendChild(h('input'));
    rng.type = 'range'; rng.min = '0'; rng.max = '1'; rng.step = '0.05'; rng.value = String(this.minConf);
    rng.oninput = () => { this.minConf = parseFloat(rng.value); cv.textContent = `Minimum confidence ${this.minConf.toFixed(2)}`; this.applyFilters(); };

    const opts = body.appendChild(h('div', 'arm-opts'));
    const check = (label, get, set) => {
      const l = opts.appendChild(h('label', 'arm-check'));
      const c = l.appendChild(h('input')); c.type = 'checkbox'; c.checked = get();
      c.onchange = () => { set(c.checked); this.dirty = true; };
      l.appendChild(h('span', null, label));
    };
    check('Show lines at rest', () => this.linesAtRest, v => { this.linesAtRest = v; });
    check('Slow drift', () => this.motion, v => { this.motion = v; });

    const btns = body.appendChild(h('div', 'arm-btns'));
    const fitBtn = btns.appendChild(h('button', null, 'Fit view')); fitBtn.onclick = () => this.fit(true);
    const reload = btns.appendChild(h('button', null, 'Reload')); reload.onclick = () => this.loadMap(false);
  }

  /* ---- right inspector */
  renderInspector() {
    const I = this.inspector;
    I.empty();
    if (!this.model || (!this.selNode && !this.selLink)) { I.addClass('arm-hidden'); return; }
    I.removeClass('arm-hidden');
    const close = I.appendChild(h('button', 'arm-icon-btn arm-close'));
    setIcon(close, 'x'); close.setAttribute('aria-label', 'Close'); close.onclick = () => this.select(null, null);
    if (this.selLink) this.renderRelation(I, this.selLink); else this.renderPaper(I, this.selNode);
  }
  paperCard(parent, n, role) {
    const card = parent.appendChild(h('div', 'arm-card'));
    card.style.setProperty('--arm-hue', this.col(this.catHue(n), 1));
    if (role) card.appendChild(h('div', 'arm-card-role', role));
    card.appendChild(h('div', 'arm-card-title', n.p.title));
    card.appendChild(h('div', 'arm-card-sub', [shortName(n.p), n.p.venue].filter(Boolean).join(' · ')));
    card.onclick = () => this.select(n, null);
    return card;
  }
  renderRelation(I, l) {
    const lab = this.model.labels[l.li];
    const chip = I.appendChild(h('div', 'arm-chip'));
    chip.style.setProperty('--arm-rel', this.relCol(l.li, 1));
    chip.appendChild(h('span', 'arm-chip-dot'));
    chip.appendChild(h('span', null, lab ? lab.label : l.r.label));
    if (lab && lab.description) I.appendChild(h('div', 'arm-muted', lab.description));
    this.paperCard(I, l.a, 'This paper');
    I.appendChild(h('div', 'arm-arrow', this.relVerb(l)));
    this.paperCard(I, l.b, 'cites');
    if (l.r.detail) { I.appendChild(h('h4', null, 'How they are related')); I.appendChild(h('p', null, l.r.detail)); }
    if (l.r.difference) { I.appendChild(h('h4', null, 'How they differ')); I.appendChild(h('p', null, l.r.difference)); }
    if (l.r.evidence && l.r.evidence.length) {
      I.appendChild(h('h4', null, 'Evidence from the text'));
      for (const e of l.r.evidence) I.appendChild(h('blockquote', null, e));
    }
    const meta = I.appendChild(h('div', 'arm-meta'));
    const c = l.r.confidence || 0;
    const bar = meta.appendChild(h('div', 'arm-bar')); const fill = bar.appendChild(h('span')); fill.style.width = `${Math.round(c * 100)}%`;
    meta.appendChild(h('span', null, `Confidence ${c.toFixed(2)}${l.r.model ? ' · ' + l.r.model : ''}`));
    if (c < 0.5) I.appendChild(h('div', 'arm-warn', 'Low confidence: the model had little to go on. Check the papers before relying on this.'));
    const btns = I.appendChild(h('div', 'arm-btns'));
    const a = btns.appendChild(h('button', null, 'Open citing note')); a.onclick = () => this.openNote(l.a);
    const b = btns.appendChild(h('button', null, 'Open cited note')); b.onclick = () => this.openNote(l.b);
  }
  relVerb(l) { const lab = this.model.labels[l.li]; return lab ? lab.label.toLowerCase() : l.r.label; }
  renderPaper(I, n) {
    const p = n.p;
    I.appendChild(h('h3', 'arm-ptitle', p.title));
    const by = [(p.authors || []).join(', ') + (p.n_authors > (p.authors || []).length ? ' et al.' : ''), p.year, p.venue].filter(Boolean).join(' · ');
    I.appendChild(h('div', 'arm-muted', by));
    const cats = I.appendChild(h('div', 'arm-cats'));
    for (const c of p.categories || []) {
      const def = this.model.catById.get(c.id); if (!def) continue;
      const top = this.model.clusters.find(x => x.id === def.top);
      const chip = cats.appendChild(h('span', 'arm-cat', def.label));
      chip.style.setProperty('--arm-hue', this.col(top ? top.hue : null, 1));
      chip.setAttribute('title', `confidence ${(c.confidence || 0).toFixed(2)}`);
    }
    if (p.rationale) I.appendChild(h('div', 'arm-muted arm-why', p.rationale));
    const groups = [['Cites', this.model.out.get(n.id) || [], l => l.b], ['Cited by', this.model.inn.get(n.id) || [], l => l.a]];
    for (const [title, ls, other] of groups) {
      const vis = ls.filter(l => l.vis);
      if (!vis.length) continue;
      I.appendChild(h('h4', null, `${title} (${vis.length})`));
      for (const l of vis) {
        const o = other(l);
        const row = I.appendChild(h('div', 'arm-rel'));
        row.style.setProperty('--arm-rel', this.relCol(l.li, 1));
        const top = row.appendChild(h('div', 'arm-rel-top'));
        top.appendChild(h('span', 'arm-chip-dot'));
        top.appendChild(h('span', 'arm-rel-label', this.model.labels[l.li] ? this.model.labels[l.li].label : l.r.label));
        row.appendChild(h('div', 'arm-rel-title', o.p.title));
        if (l.r.detail) row.appendChild(h('div', 'arm-rel-detail', l.r.detail));
        row.onclick = () => this.select(null, l);
        row.onmouseenter = () => { this.hoverLink = l; this.dirty = true; };
        row.onmouseleave = () => { this.hoverLink = null; this.dirty = true; };
      }
    }
    if (p.abstract) {
      I.appendChild(h('h4', null, 'Abstract'));
      I.appendChild(h('p', 'arm-abstract', p.abstract));
    }
    const btns = I.appendChild(h('div', 'arm-btns'));
    const o = btns.appendChild(h('button', 'mod-cta', 'Open note')); o.onclick = () => this.openNote(n);
    if (p.url) { const w = btns.appendChild(h('button', null, 'Open page')); w.onclick = () => window.open(p.url); }
  }
}

function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  const t = l2 ? clamp(((px - ax) * dx + (py - ay) * dy) / l2, 0, 1) : 0;
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/* ------------------------------------------------------------------ plugin */
module.exports = class AutorefGraph extends Plugin {
  async onload() {
    dbg('plugin loaded');
    this.registerView(VIEW, leaf => new PaperMapView(leaf, this));
    this.addRibbonIcon('orbit', 'Open paper map', () => this.openMap());
    this.addCommand({ id: 'open-paper-map', name: 'Open paper map', callback: () => this.openMap() });
  }
  onunload() { this.app.workspace.detachLeavesOfType(VIEW); }
  async openMap() {
    const existing = this.app.workspace.getLeavesOfType(VIEW)[0];
    if (existing) { this.app.workspace.revealLeaf(existing); return; }
    const leaf = this.app.workspace.getLeaf('tab');
    await leaf.setViewState({ type: VIEW, active: true });
    this.app.workspace.revealLeaf(leaf);
  }
};
