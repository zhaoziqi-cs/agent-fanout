/* =============================================================================
 *  topology.js —— 调研拓扑渲染（原生 JS + SVG，零依赖）
 * =============================================================================
 *  一颗主节点 + N 个 subagent：主节点画成方形（派发控制台），subagent 画成
 *  圆形（工作节点）——角色由形状区分，颜色只表达状态：
 *
 *      等待 #3A4249（灰）  →  运行 #E3A33C（琥珀，带脉冲光环）  →  完成 #5FA98A
 *
 *  连线表示一次派发，随子节点状态同步变色；运行时节点的脉冲用 SMIL 声明式
 *  动画，尊重 prefers-reduced-motion。
 *
 *  几何：viewBox 宽 360、高随实际用到的行数收缩，SVG 宽度 100% 随容器缩放，
 *  因此子节点少时不会在底部留出空白。最多 6 个子节点（2 列 × 3 行），
 *  与 agents.py 的单次派发上限一致。
 * =========================================================================== */

const NS = 'http://www.w3.org/2000/svg';

const C = {
  idle:  '#454F57',   // 等待
  live:  '#E3A33C',   // 运行
  done:  '#5FA98A',   // 完成
  edge:  '#2A3238',   // 未激活的连线
  fill:  '#161C21',   // 节点底色
  label: '#6E7A82',   // 节点文字
};

const W = 360;                   // viewBox 宽度
const MAIN = { x: 180, y: 40, w: 88, h: 30, rx: 6 };
const COL_X = [95, 265];         // 子节点两列
const ROW_Y = [138, 232, 326];   // 子节点三行
const R = 17;                    // 子节点半径

class TopoViz {
  constructor(host) {
    this.host = host;
    this.host.innerHTML = '';          // ← 切换问题整体换图，不堆叠
    this.subs = {};
    this.order = [];
    this._clickCb = null;
    this.reduceMotion = !!(window.matchMedia &&
      window.matchMedia('(prefers-reduced-motion: reduce)').matches);
    this.svg = document.createElementNS(NS, 'svg');
    this.svg.setAttribute('width', '100%');
    this.svg.setAttribute('role', 'img');
    this.svg.setAttribute('aria-label', '调研拓扑：主 Agent 与各 subagent 的派发关系');
    // 发光滤镜：仅在节点处于「运行」状态时挂上，静止的节点保持安静
    const defs = document.createElementNS(NS, 'defs');
    defs.innerHTML = `
      <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="3" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>`;
    this.svg.appendChild(defs);
    this.layers = { edges: document.createElementNS(NS, 'g'),
                    nodes: document.createElementNS(NS, 'g') };
    this.svg.appendChild(this.layers.edges);
    this.svg.appendChild(this.layers.nodes);
    this.host.appendChild(this.svg);
  }

  /* 节点：shape 为圆或方，整组可点击（点节点即筛选该 agent 的步骤流） */
  _node(shape, x, y, label, id, stroke) {
    const g = document.createElementNS(NS, 'g');
    g.style.cursor = 'pointer';
    shape.style.fill = C.fill;
    shape.style.stroke = stroke;
    shape.style.strokeWidth = '2';
    shape.style.transition = 'stroke .3s, fill .3s';
    g.appendChild(shape);

    // 主节点文字压在方块中心，子节点文字排在圆下方
    const isMain = shape.tagName === 'rect';
    const t = document.createElementNS(NS, 'text');
    t.setAttribute('text-anchor', 'middle');
    t.setAttribute('font-size', isMain ? '10.5' : '9.5');
    t.setAttribute('x', x);
    t.setAttribute('y', isMain ? y + 3.5 : y + R + 18);
    t.style.fill = C.label;
    t.style.transition = 'fill .3s';
    t.textContent = label;
    const full = document.createElementNS(NS, 'title');
    full.textContent = label;
    g.appendChild(full);
    g.appendChild(t);
    if (id) g.addEventListener('click', () => this._clickCb && this._clickCb(id));
    this.layers.nodes.appendChild(g);
    return { g, shape, t };
  }

  /* 主 → 子 的贝塞尔连线，起点在方块下沿，终点在圆的上沿 */
  _edge(x2, y2) {
    const y1 = MAIN.y + MAIN.h / 2, mid = (y1 + (y2 - R)) / 2;
    const p = document.createElementNS(NS, 'path');
    p.setAttribute('d', `M${MAIN.x},${y1} C${MAIN.x},${mid} ${x2},${mid} ${x2},${y2 - R}`);
    p.setAttribute('fill', 'none');
    p.style.stroke = C.edge;
    p.style.strokeWidth = '1.5';
    p.style.strokeDasharray = '3 4';
    p.style.transition = 'stroke .3s';
    this.layers.edges.appendChild(p);
    return p;
  }

  /* 按实际用到的行数收缩 viewBox 高度：子节点少时不至于在下方留出大片空白。
     尚未派发时只框住主节点，卡片保持紧凑，随派发逐行长高。 */
  _fit() {
    const rows = Math.ceil(this.order.length / COL_X.length);
    const bottom = rows === 0
      ? MAIN.y + MAIN.h / 2 + 20
      : ROW_Y[Math.min(rows, ROW_Y.length) - 1] + R + 29;
    this.svg.setAttribute('viewBox', `0 0 ${W} ${bottom}`);
  }

  /* 同一形状的描边光环：描边变粗、透明度归零，循环即脉冲。
     关掉动效偏好时退化为静态描边，不再闪动。 */
  _halo(shape, color) {
    const h = shape.cloneNode(false);
    h.removeAttribute('filter');
    h.style.fill = 'none';
    h.style.stroke = color;
    h.style.strokeDasharray = 'none';
    h.style.pointerEvents = 'none';
    const pulse = (name, values) => {
      const a = document.createElementNS(NS, 'animate');
      a.setAttribute('attributeName', name);
      a.setAttribute('values', values);
      a.setAttribute('dur', '1.5s');
      a.setAttribute('repeatCount', 'indefinite');
      h.appendChild(a);
    };
    if (this.reduceMotion) {
      h.style.strokeWidth = '3';
      h.style.strokeOpacity = '0.35';
    } else {
      h.style.strokeWidth = '1.5';
      pulse('stroke-opacity', '0.75;0');
      pulse('stroke-width', '1.5;7');
    }
    return h;
  }

  setMain() {
    const r = document.createElementNS(NS, 'rect');
    r.setAttribute('x', MAIN.x - MAIN.w / 2);
    r.setAttribute('y', MAIN.y - MAIN.h / 2);
    r.setAttribute('width', MAIN.w);
    r.setAttribute('height', MAIN.h);
    r.setAttribute('rx', MAIN.rx);
    const o = this._node(r, MAIN.x, MAIN.y, '主 Agent', 'main', C.idle);
    this.subs['main'] = { ...o, x: MAIN.x, y: MAIN.y, kind: 'rect', status: 'idle' };
    this._fit();
  }

  addSubagent(id, topic) {
    const i = this.order.length;
    const x = COL_X[i % 2], y = ROW_Y[Math.min(Math.floor(i / 2), ROW_Y.length - 1)];
    const c = document.createElementNS(NS, 'circle');
    c.setAttribute('cx', x); c.setAttribute('cy', y); c.setAttribute('r', R);
    const label = topic.length > 12 ? topic.slice(0, 12) + '…' : topic;
    const o = this._node(c, x, y, label, id, C.idle);
    const edge = this._edge(x, y);
    this.subs[id] = { ...o, x, y, kind: 'circle', status: 'idle', topic, edge };
    this.order.push(id);
    this._fit();
  }

  markRunning(id) {
    const s = this.subs[id]; if (!s) return;
    s.status = 'running';
    s.shape.style.fill = '#3A2A0C';
    s.shape.style.stroke = C.live;
    s.shape.style.strokeWidth = '2.5';
    s.shape.setAttribute('filter', 'url(#glow)');   // 只有运行中的节点发光
    s.t.style.fill = C.live;
    if (!s.halo) {
      s.halo = this._halo(s.shape, C.live);
      s.g.insertBefore(s.halo, s.g.firstChild);
    }
    if (s.edge) { s.edge.style.stroke = C.live; s.edge.style.strokeDasharray = 'none'; }
  }

  markDone(id) {
    const s = this.subs[id]; if (!s) return;
    s.status = 'done';
    if (s.halo) { s.halo.remove(); s.halo = null; }
    s.shape.style.fill = '#12241E';
    s.shape.style.stroke = C.done;
    s.shape.style.strokeWidth = '2';
    s.shape.removeAttribute('filter');
    s.t.style.fill = id === 'main' ? C.done : '#8AA79A';
    if (s.edge) { s.edge.style.stroke = C.done; s.edge.style.strokeDasharray = 'none'; }
  }

  reset() {
    Object.values(this.subs).forEach(s => { if (s.halo) { s.halo.remove(); s.halo = null; } });
    this.host.innerHTML = '';
    this.subs = {}; this.order = [];
    this.svg = null;                  // 下次 new TopoViz 重新建图
  }

  onClick(cb) { this._clickCb = cb; }
}
