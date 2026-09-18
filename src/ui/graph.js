// Inter-basin transition graph. Nodes are catalog states laid out by energy
// (lowest at the bottom-left, highest at the top-right); edges are the escape
// targets found by the basin bisections, width ∝ count, coloured and labelled
// by the energy climbed along the ray (ΔU). Pseudo-nodes collect escapes into
// uncatalogued minima ("new") and onto saddles. No physics lives here.

const PAD = 30;
const TOP = 46;

const lerp = (a, b, t) => a + (b - a) * t;

/** Low barrier → amber (easy transition), high barrier → violet; unknown → grey. */
function barrierColor(barrier, bMin, bMax, alpha) {
    if (barrier === null || barrier === undefined) return `rgba(140,150,170,${alpha})`;
    const t = bMax > bMin ? (barrier - bMin) / (bMax - bMin) : 0.5;
    const r = Math.round(lerp(255, 157, t)),
        g = Math.round(lerp(181, 140, t)),
        b = Math.round(lerp(71, 255, t));
    return `rgba(${r},${g},${b},${alpha})`;
}

function nodeRadius(share) {
    return 11 + (share > 0 ? 9 * Math.sqrt(Math.min(1, share)) : 0);
}

/**
 * Compute node positions. Returns a Map id -> {id, x, y, r, node, virtual, label}.
 * `shares` (Map id -> basin fraction) scales the node radius when given.
 */
export function layoutGraph(graph, width, height, shares = null) {
    const nodes = graph.nodes.slice().sort((a, b) => a.energy - b.energy);
    const n = nodes.length;
    const layout = new Map();
    if (!n) return layout;
    const Emin = nodes[0].energy,
        Emax = nodes[n - 1].energy;
    const span = height - PAD - TOP;
    nodes.forEach((nd, k) => {
        const t = Emax > Emin ? (nd.energy - Emin) / (Emax - Emin) : 0.5;
        const x = PAD + ((k + 0.5) / n) * (width - 2 * PAD);
        const y = height - PAD - t * span;
        const share = shares ? shares.get(nd.id) || 0 : 0;
        layout.set(nd.id, {id: nd.id, x, y, r: nodeRadius(share), node: nd, virtual: false});
    });
    if (graph.edges.some((e) => e.to === 0)) {
        layout.set(0, {id: 0, x: width - PAD, y: 18, r: 11, virtual: true, label: 'new'});
    }
    if (graph.edges.some((e) => e.to === -1)) {
        layout.set(-1, {id: -1, x: PAD + 6, y: 18, r: 15, virtual: true, label: 'saddle'});
    }
    return layout;
}

function arrow(ctx, x, y, ux, uy, size) {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - ux * size - uy * size * 0.5, y - uy * size + ux * size * 0.5);
    ctx.lineTo(x - ux * size + uy * size * 0.5, y - uy * size - ux * size * 0.5);
    ctx.closePath();
    ctx.fill();
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

function pill(ctx, text, x, y) {
    ctx.font = '10px ui-monospace, Menlo, Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const w = ctx.measureText(text).width + 8,
        h = 14;
    ctx.fillStyle = 'rgba(13,16,24,0.88)';
    roundRect(ctx, x - w / 2, y - h / 2, w, h, 4);
    ctx.fill();
    ctx.fillStyle = '#d5dae6';
    ctx.fillText(text, x, y);
}

/**
 * Render the transition graph. opts = { selected: id, hover: id|null, shares: Map }.
 * Returns the layout used (for hit-testing with graphNodeAt).
 */
export function renderTransitionGraph(canvas, graph, opts = {}) {
    const {selected = -1, hover = null, shares = null} = opts;
    const ctx = canvas.getContext('2d');
    const W = canvas.width,
        H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    const layout = layoutGraph(graph, W, H, shares);
    if (!layout.size) return layout;

    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = '#5c6478';
    ctx.fillText('higher energy ↑', W - 6, H - 6);

    let maxCount = 1,
        bMin = Infinity,
        bMax = -Infinity;
    for (const e of graph.edges) {
        maxCount = Math.max(maxCount, e.count);
        if (e.barrier !== null) {
            bMin = Math.min(bMin, e.barrier);
            bMax = Math.max(bMax, e.barrier);
        }
    }
    const focus = hover !== null ? hover : selected > 0 ? selected : null;
    const labelAll = graph.edges.length <= 10;

    // edges
    ctx.lineCap = 'round';
    const labels = [];
    for (const e of graph.edges) {
        const a = layout.get(e.from),
            b = layout.get(e.to);
        if (!a || !b) continue;
        const touches = focus !== null && (e.from === focus || e.to === focus);
        const dim = focus !== null && !touches;
        const alpha = dim ? 0.15 : 0.85;
        const color = barrierColor(e.barrier, bMin, bMax, alpha);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 1 + 3.5 * (e.count / maxCount);
        let lx, ly;
        if (a === b) {
            // self-transition (e.g. onto a symmetry image of the same state)
            const cy = a.y - a.r - 8;
            ctx.beginPath();
            ctx.arc(a.x, cy, 8, 0, Math.PI * 2);
            ctx.stroke();
            lx = a.x;
            ly = cy - 17;
        } else {
            const dx = b.x - a.x,
                dy = b.y - a.y,
                len = Math.hypot(dx, dy) || 1;
            const nx = -dy / len,
                ny = dx / len;
            const bend = Math.min(0.25 * len, 34);
            const cx = (a.x + b.x) / 2 + nx * bend,
                cy = (a.y + b.y) / 2 + ny * bend;
            const s1x = cx - a.x,
                s1y = cy - a.y,
                l1 = Math.hypot(s1x, s1y) || 1;
            const sx = a.x + (s1x / l1) * (a.r + 2),
                sy = a.y + (s1y / l1) * (a.r + 2);
            const e1x = b.x - cx,
                e1y = b.y - cy,
                l2 = Math.hypot(e1x, e1y) || 1;
            const ex = b.x - (e1x / l2) * (b.r + 3),
                ey = b.y - (e1y / l2) * (b.r + 3);
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.quadraticCurveTo(cx, cy, ex, ey);
            ctx.stroke();
            arrow(ctx, ex, ey, e1x / l2, e1y / l2, 5 + ctx.lineWidth);
            lx = 0.25 * sx + 0.5 * cx + 0.25 * ex;
            ly = 0.25 * sy + 0.5 * cy + 0.25 * ey;
        }
        if (!dim && (labelAll || touches)) {
            const parts = [];
            if (e.barrier !== null) parts.push(`ΔU ${e.barrier.toFixed(2)}`);
            if (e.count > 1 || e.barrier === null) parts.push(`×${e.count}`);
            labels.push({text: parts.join(' '), x: lx, y: ly});
        }
    }
    ctx.lineCap = 'butt';

    // nodes
    for (const nd of layout.values()) {
        const isSel = nd.id === selected,
            isHov = nd.id === hover;
        ctx.beginPath();
        ctx.arc(nd.x, nd.y, nd.r, 0, Math.PI * 2);
        if (nd.virtual) {
            ctx.fillStyle = 'rgba(20,24,34,0.92)';
            ctx.fill();
            ctx.setLineDash([3, 3]);
            ctx.lineWidth = isHov ? 1.8 : 1.2;
            ctx.strokeStyle = nd.id === 0 ? '#5fd0c0' : '#d07a7a';
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = nd.id === 0 ? '#9fe6dc' : '#f0b0b0';
            ctx.font = '9px system-ui, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(nd.label, nd.x, nd.y);
            continue;
        }
        const saddle = nd.node.saddle;
        if (isSel) {
            ctx.shadowColor = 'rgba(79,209,165,0.65)';
            ctx.shadowBlur = 14;
        }
        const grad = ctx.createRadialGradient(
            nd.x - nd.r * 0.3,
            nd.y - nd.r * 0.35,
            1,
            nd.x,
            nd.y,
            nd.r,
        );
        grad.addColorStop(0, saddle ? '#8a5c66' : '#4f92da');
        grad.addColorStop(1, saddle ? '#4d2f38' : '#234c7c');
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.lineWidth = isSel ? 2.5 : 1.5;
        ctx.strokeStyle = isSel ? '#4fd1a5' : isHov ? '#e6ecf8' : '#7fa9d8';
        ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 11px ui-monospace, Menlo, Consolas, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`#${nd.id}`, nd.x, nd.y);
        ctx.fillStyle = '#8d95a9';
        ctx.font = '9px ui-monospace, Menlo, Consolas, monospace';
        ctx.fillText(`U ${nd.node.energy.toFixed(2)}`, nd.x, nd.y + nd.r + 8);
    }

    // labels last so they stay legible on top of everything
    for (const l of labels) pill(ctx, l.text, l.x, l.y);
    ctx.textAlign = 'start';
    ctx.textBaseline = 'alphabetic';
    return layout;
}

/** Map a point (CSS px relative to the canvas) to a node id (0 / -1 for the pseudo-nodes), or null. */
export function graphNodeAt(canvas, layout, cssX, cssY) {
    if (!layout) return null;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    const x = (cssX * canvas.width) / rect.width;
    const y = (cssY * canvas.height) / rect.height;
    let best = null,
        bd = Infinity;
    for (const nd of layout.values()) {
        const d = Math.hypot(x - nd.x, y - nd.y);
        if (d <= nd.r + 3 && d < bd) {
            bd = d;
            best = nd.id;
        }
    }
    return best;
}
