// 2D rendering: grid, dipole disks, orientation arrows, plus a pan/zoom camera.
// No physics lives here.

export class SceneRenderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        // Camera: world origin sits at canvas centre + offset (CSS px); zoom scales world->screen.
        this.offset = [0, 0];
        this.zoom = 1;
        this.minZoom = 0.1;
        this.maxZoom = 10;
        this.dpr = 1;
        this.w = 0;
        this.h = 0;
        /** Optional callback invoked after the backing store is resized. */
        this.onResize = null;

        this.resize();
        if (typeof ResizeObserver !== 'undefined') {
            this._ro = new ResizeObserver(() => this.resize());
            this._ro.observe(canvas);
        } else {
            window.addEventListener('resize', () => this.resize());
        }
    }

    /** Sync the backing store with the element's CSS size (guarded against no-op loops). */
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = Math.max(1, Math.round(rect.width));
        const h = Math.max(1, Math.round(rect.height));
        if (w === this.w && h === this.h && dpr === this.dpr) return;
        this.w = w;
        this.h = h;
        this.dpr = dpr;
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
        if (this.onResize) this.onResize();
    }

    /** Screen (canvas-local CSS px) -> world (px at zoom 1). */
    screenToWorld(sx, sy) {
        return [
            (sx - this.w / 2 - this.offset[0]) / this.zoom,
            (sy - this.h / 2 - this.offset[1]) / this.zoom,
        ];
    }

    worldToScreen(wx, wy) {
        return [
            wx * this.zoom + this.w / 2 + this.offset[0],
            wy * this.zoom + this.h / 2 + this.offset[1],
        ];
    }

    panBy(dx, dy) {
        this.offset[0] += dx;
        this.offset[1] += dy;
    }

    /** Zoom by `factor` keeping the world point under screen (sx, sy) fixed. */
    zoomAt(sx, sy, factor) {
        const [wx, wy] = this.screenToWorld(sx, sy);
        this.zoom = Math.min(this.maxZoom, Math.max(this.minZoom, this.zoom * factor));
        this.offset[0] = sx - this.w / 2 - wx * this.zoom;
        this.offset[1] = sy - this.h / 2 - wy * this.zoom;
    }

    resetView() {
        this.offset = [0, 0];
        this.zoom = 1;
    }

    /**
     * Render the lattice.
     * opts.angles    — optional Float64Array overriding magnet angles (mode preview)
     * opts.modeVec   — optional eigenvector to overlay as wedges
     * opts.hover     — index of hovered magnet (or -1)
      * opts.highlight — optional array of magnet indices to ring (coupling-matrix hover)
     * opts.coupling  — optional n×n coupling matrix (normalised Hessian C) drawn as lines
     *                  between the cores: red C_ij > 0, blue C_ij < 0, opacity/width ∝ |C_ij|
     * opts.showLabels— draw magnet ids
     */
    render(lattice, opts = {}) {
        const {
            angles = null,
            modeVec = null,
            hover = -1,
            highlight = null,
            coupling = null,
            showLabels = true,
        } = opts;
        const ctx = this.ctx;
        ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
        ctx.clearRect(0, 0, this.w, this.h);

        const pitch = lattice.pitch;
        const z = this.zoom;
        const extent = lattice.extent;

        // Visible range in grid cells.
        const [wx0, wy0] = this.screenToWorld(0, 0);
        const [wx1, wy1] = this.screenToWorld(this.w, this.h);
        let gx0 = Math.floor(wx0 / pitch),
            gx1 = Math.ceil(wx1 / pitch);
        let gy0 = Math.floor(wy0 / pitch),
            gy1 = Math.ceil(wy1 / pitch);
        if (extent !== null) {
            gx0 = Math.max(gx0, -extent);
            gx1 = Math.min(gx1, extent);
            gy0 = Math.max(gy0, -extent);
            gy1 = Math.min(gy1, extent);
        }

        // Grid boundary (bounded lattices).
        if (extent !== null) {
            const half = (extent + 0.5) * pitch;
            const [ax, ay] = this.worldToScreen(-half, -half);
            const [bx, by] = this.worldToScreen(half, half);
            ctx.strokeStyle = '#3a3a4a';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.strokeRect(ax, ay, bx - ax, by - ay);
            ctx.setLineDash([]);
        }

        // Grid dots (skipped when too dense to be useful).
        const sp = pitch * z;
        if (sp >= 5 && (gx1 - gx0 + 1) * (gy1 - gy0 + 1) <= 40000) {
            ctx.fillStyle = '#2a2a33';
            const r = Math.min(1.5, sp * 0.1);
            for (let gx = gx0; gx <= gx1; gx++) {
                for (let gy = gy0; gy <= gy1; gy++) {
                    const [sx, sy] = this.worldToScreen(gx * pitch, gy * pitch);
                    ctx.beginPath();
                    ctx.arc(sx, sy, r, 0, Math.PI * 2);
                    ctx.fill();
                }
            }
        }

        // Origin marker.
        {
            const [ox, oy] = this.worldToScreen(0, 0);
            ctx.strokeStyle = '#33334a';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(ox - 6, oy);
            ctx.lineTo(ox + 6, oy);
            ctx.moveTo(ox, oy - 6);
            ctx.lineTo(ox, oy + 6);
            ctx.stroke();
        }

        const R = pitch * 0.36 * z;
        // Coupling lines (analysis mode). Drawn under the magnets, disk-edge to disk-edge.
        if (coupling) {
            const n = Math.min(coupling.length, lattice.magnets.length);
            const centres = new Array(n);
            for (let i = 0; i < n; i++) {
                const [wx, wy] = lattice.cellToWorld(lattice.magnets[i].cell);
                centres[i] = this.worldToScreen(wx, wy);
            }
            const hl = highlight && highlight.length === 2 && highlight[0] !== highlight[1] ? highlight : null;
            ctx.lineCap = 'round';
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    const v = coupling[i][j];
                    const a = Math.min(1, Math.abs(v));
                    const isHl = hl && ((hl[0] === i && hl[1] === j) || (hl[0] === j && hl[1] === i));
                    if (a < 0.03 && !isHl) continue; // far pairs (1/r³) would only add clutter
                    const [x0, y0] = centres[i],
                        [x1, y1] = centres[j];
                    const dx = x1 - x0,
                        dy = y1 - y0;
                    const len = Math.hypot(dx, dy);
                    if (len <= 2 * R) continue;
                    const ux = dx / len,
                        uy = dy / len;
                    ctx.beginPath();
                    ctx.moveTo(x0 + ux * R, y0 + uy * R);
                    ctx.lineTo(x1 - ux * R, y1 - uy * R);
                    if (isHl) {
                        ctx.strokeStyle = '#fc6';
                        ctx.lineWidth = 3;
                    } else {
                        const alpha = 0.15 + 0.75 * a;
                        ctx.strokeStyle = v > 0 ? `rgba(230,90,90,${alpha})` : `rgba(90,130,230,${alpha})`;
                        ctx.lineWidth = 0.75 + 3 * a;
                    }
                    ctx.stroke();
                }
            }
            ctx.lineCap = 'butt';
        }
        // Magnets.
        const lw = Math.max(1, Math.min(3, R * 0.14));
        const headLen = Math.min(8, R * 0.5);
        const margin = R + 24;
        lattice.magnets.forEach((mg, idx) => {
            const [wx, wy] = lattice.cellToWorld(mg.cell);
            const [sx, sy] = this.worldToScreen(wx, wy);
            if (sx < -margin || sy < -margin || sx > this.w + margin || sy > this.h + margin) return;

            const th = angles ? angles[idx] : mg.theta;
            const ex = Math.cos(th),
                ey = Math.sin(th);

            // hover ring
            if (idx === hover) {
                ctx.beginPath();
                ctx.arc(sx, sy, R + 4, 0, Math.PI * 2);
                ctx.strokeStyle = '#8ac';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
             // coupling highlight ring
             if (highlight && highlight.includes(idx)) {
                 ctx.beginPath();
                 ctx.arc(sx, sy, R + 8, 0, Math.PI * 2);
                 ctx.strokeStyle = '#fc6';
                 ctx.lineWidth = 2.5;
                 ctx.stroke();
             }

            // disk
            ctx.beginPath();
            ctx.arc(sx, sy, R, 0, Math.PI * 2);
            ctx.fillStyle = '#2c3540';
            ctx.fill();
            ctx.lineWidth = Math.max(1, lw * 0.5);
            ctx.strokeStyle = '#4a6a8a';
            ctx.stroke();

            // dipole: north half red, south half blue
            ctx.lineWidth = lw;
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.lineTo(sx + ex * R, sy + ey * R);
            ctx.strokeStyle = '#e05555';
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.lineTo(sx - ex * R, sy - ey * R);
            ctx.strokeStyle = '#5577e0';
            ctx.stroke();

            // arrowhead
            if (headLen > 2) {
                const hx = sx + ex * R,
                    hy = sy + ey * R;
                const a = 0.5;
                ctx.beginPath();
                ctx.moveTo(hx, hy);
                ctx.lineTo(
                    hx - (ex * Math.cos(a) - ey * Math.sin(a)) * headLen,
                    hy - (ex * Math.sin(a) + ey * Math.cos(a)) * headLen,
                );
                ctx.moveTo(hx, hy);
                ctx.lineTo(
                    hx - (ex * Math.cos(-a) - ey * Math.sin(-a)) * headLen,
                    hy - (ex * Math.sin(-a) + ey * Math.cos(-a)) * headLen,
                );
                ctx.strokeStyle = '#e05555';
                ctx.lineWidth = Math.max(1, lw * 0.7);
                ctx.stroke();
            }

            // mode overlay: eigenvector component as a coloured wedge
            if (modeVec && idx < modeVec.length) {
                const amp = modeVec[idx];
                ctx.beginPath();
                ctx.arc(sx, sy, R + 5, th - 0.4, th - 0.4 + amp * 1.5, amp < 0);
                ctx.strokeStyle = amp >= 0 ? '#6f6' : '#f66';
                ctx.lineWidth = 2;
                ctx.stroke();
            }

            // id label
            if (showLabels && R > 6) {
                ctx.fillStyle = '#889';
                ctx.font = '10px monospace';
                ctx.fillText(String(mg.id), sx + R + 2, sy - R);
            }
        });
    }
}