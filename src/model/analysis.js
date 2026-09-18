// Equilibrium finding, normal modes, coupling matrix, lattice symmetries,
// stable-state sweep, basin-of-attraction analysis and the transition graph.

import {energy, gradient, hessian} from './physics.js';
import {jacobiEigen, luSolve} from '../math/linalg.js';

function norm(v) {
    let s = 0;
    for (let i = 0; i < v.length; i++) s += v[i] * v[i];
    return Math.sqrt(s);
}

/**
 * In-place gradient descent with Armijo backtracking.
 * Stops when |∇U| < tol, after maxIter iterations, or when the step size
 * underflows (the energy can no longer be resolved in floating point).
 */
function descend(theta, pairs, n, maxIter, tol) {
    let alpha = 0.1;
    let iterations = 0;
    let gradNorm = Infinity;
    for (iterations = 0; iterations < maxIter; iterations++) {
        const g = gradient(theta, pairs, n);
        gradNorm = norm(g);
        if (gradNorm < tol) break;

        const E0 = energy(theta, pairs);
        let s = alpha;
        let accepted = false;
        const trial = new Float64Array(n);
        for (let ls = 0; ls < 30; ls++) {
            for (let i = 0; i < n; i++) trial[i] = theta[i] - s * g[i];
            const E1 = energy(trial, pairs);
            if (E1 < E0 - 1e-4 * s * gradNorm * gradNorm) {
                theta.set(trial);
                alpha = s * 1.5;
                accepted = true;
                break;
            }
            s *= 0.5;
        }
        if (!accepted) {
            alpha *= 0.5;
            if (alpha < 1e-14) break;
        }
    }
    return {iterations, gradNorm};
}

/**
 * Relax to equilibrium.
 *
 * Phase 1: gradient descent (robust, finds the basin).
 * Phase 2: Newton polish with the analytic Hessian. Pure gradient descent
 *          stalls around |∇U| ~ 1e-8 because energy differences fall below
 *          floating-point resolution; Newton converges quadratically to ~1e-15.
 *          Steps are only accepted along descent directions that shrink |∇U|
 *          without raising U, so a saddle cannot be "polished into".
 * Phase 3: fall back to gradient descent at the fine tolerance if Newton bailed.
 *
 * Returns { theta, iterations, gradNorm }.
 */
export function relax(theta0, pairs, n, opts = {}) {
    const {maxIter = 2000, tol = 1e-9, newtonIter = 25, coarseTol = 1e-4} = opts;
    const theta = Float64Array.from(theta0);
    let iterations = 0;

    let r = descend(theta, pairs, n, maxIter, Math.max(coarseTol, tol));
    iterations += r.iterations;
    let gradNorm = r.gradNorm;

    for (let it = 0; it < newtonIter && gradNorm > tol; it++) {
        const g = gradient(theta, pairs, n);
        const H = hessian(theta, pairs, n);
        const rhs = new Float64Array(n);
        for (let i = 0; i < n; i++) rhs[i] = -g[i];
        let d;
        try {
            d = luSolve(H, rhs);
        } catch {
            break; // singular Hessian
        }
        let slope = 0;
        for (let i = 0; i < n; i++) slope += g[i] * d[i];
        if (!(slope < 0)) break; // not a descent direction (indefinite Hessian)

        const trial = new Float64Array(n);
        for (let i = 0; i < n; i++) trial[i] = theta[i] + d[i];
        const gn = norm(gradient(trial, pairs, n));
        const E0 = energy(theta, pairs);
        const E1 = energy(trial, pairs);
        if (!(gn < gradNorm) || E1 > E0 + 1e-12 * Math.max(1, Math.abs(E0))) break;
        theta.set(trial);
        gradNorm = gn;
        iterations++;
    }

    if (gradNorm > tol) {
        r = descend(theta, pairs, n, maxIter, tol);
        iterations += r.iterations;
        gradNorm = r.gradNorm;
    }

    // wrap into [-π, π]
    for (let i = 0; i < n; i++) theta[i] = Math.atan2(Math.sin(theta[i]), Math.cos(theta[i]));
    return {theta, iterations, gradNorm};
}

/**
 * Normal-mode analysis at (or near) equilibrium.
 * Generalized eigenproblem H v = ω² M v with M = I·Identity, so ω² = eig(H)/I.
 * Returns { H, C, values, omega2, omega, modes }.
 * `omega[i]` is negative (−sqrt|ω²|) when the mode is unstable.
 */
export function analyze(theta, pairs, n, params) {
    const {I} = params;
    const H = hessian(theta, pairs, n);
    const {values, vectors} = jacobiEigen(H);

    const omega2 = new Float64Array(n);
    const omega = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        omega2[i] = values[i] / I;
        omega[i] = omega2[i] >= 0 ? Math.sqrt(omega2[i]) : -Math.sqrt(-omega2[i]);
    }

    // Normalized coupling matrix C_ij = H_ij / sqrt(|H_ii H_jj|)
    const C = new Array(n);
    for (let i = 0; i < n; i++) {
        C[i] = new Float64Array(n);
        for (let j = 0; j < n; j++) {
            const denom = Math.sqrt(Math.abs(H[i][i] * H[j][j]));
            C[i][j] = denom > 1e-30 ? H[i][j] / denom : 0;
        }
    }

    return {H, C, values, omega2, omega, modes: vectors};
}

// ---- Symmetries -------------------------------------------------------------
//
// Two configurations are treated as the same "state" when one maps onto the
// other by a symmetry of the energy:
//   * a point-group operation of the square lattice (rotation by a multiple of
//     90° or a reflection about the centroid) that maps the magnet position set
//     onto itself — positions are permuted and every dipole vector is rotated /
//     reflected along with them; the dipole energy only depends on dot products
//     so it is invariant;
//   * the global dipole flip θ_i → θ_i + π for all i (U is bilinear in dipoles).
function wrapAngle(a) {
    return Math.atan2(Math.sin(a), Math.cos(a));
}
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
/** D4 as 2×2 matrices [a, b, c, d]: (x, y) -> (a x + b y, c x + d y). */
const D4 = [
    {name: 'e', M: [1, 0, 0, 1]},
    {name: 'r90', M: [0, -1, 1, 0]},
    {name: 'r180', M: [-1, 0, 0, -1]},
    {name: 'r270', M: [0, 1, -1, 0]},
    {name: 'mx', M: [1, 0, 0, -1]}, // reflect across the x-axis (y -> -y)
    {name: 'my', M: [-1, 0, 0, 1]}, // reflect across the y-axis
    {name: 'md', M: [0, 1, 1, 0]}, // reflect across y = x
    {name: 'md2', M: [0, -1, -1, 0]}, // reflect across y = -x
];
/**
 * Point-group operations (about the centroid) that map the position set onto
 * itself. Each op is { name, M, perm } where magnet i is sent to perm[i].
 */
export function latticeSymmetries(positions, tol = 1e-6) {
    const n = positions.length;
    if (n === 0) return [];
    let cx = 0,
        cy = 0;
    for (const p of positions) {
        cx += p[0];
        cy += p[1];
    }
    cx /= n;
    cy /= n;
    const ops = [];
    for (const {name, M} of D4) {
        const [a, b, c, d] = M;
        const perm = new Int32Array(n);
        const used = new Uint8Array(n);
        let ok = true;
        for (let i = 0; i < n && ok; i++) {
            const x = positions[i][0] - cx,
                y = positions[i][1] - cy;
            const tx = a * x + b * y + cx,
                ty = c * x + d * y + cy;
            let found = -1;
            for (let j = 0; j < n; j++) {
                if (
                    !used[j] &&
                    Math.abs(positions[j][0] - tx) < tol &&
                    Math.abs(positions[j][1] - ty) < tol
                ) {
                    found = j;
                    break;
                }
            }
            if (found < 0) ok = false;
            else {
                perm[i] = found;
                used[found] = 1;
            }
        }
        if (ok) ops.push({name, M, perm});
    }
    return ops;
}
/** Apply a point-group op to a configuration: dipoles are permuted and transformed by M. */
export function transformState(theta, op) {
    const [a, b, c, d] = op.M;
    const n = theta.length;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        const cs = Math.cos(theta[i]),
            sn = Math.sin(theta[i]);
        out[op.perm[i]] = Math.atan2(c * cs + d * sn, a * cs + b * sn);
    }
    return out;
}
/** All images of θ under the point group extended by the global dipole flip. */
export function symmetryImages(theta, ops) {
    const n = theta.length;
    const out = [];
    for (const op of ops) {
        const img = transformState(theta, op);
        out.push(img);
        const flip = new Float64Array(n);
        for (let i = 0; i < n; i++) flip[i] = wrapAngle(img[i] + Math.PI);
        out.push(flip);
    }
    return out;
}
function maxAngleDiff(a, b) {
    let m = 0;
    for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(wrapAngle(a[i] - b[i])));
    return m;
}
/** True when some symmetry image of `a` coincides with `b` within `tol` (radians). */
export function statesEquivalent(a, b, ops, tol = 1e-4) {
    if (a.length !== b.length) return false;
    for (const img of symmetryImages(a, ops)) if (maxAngleDiff(img, b) < tol) return true;
    return false;
}
/**
 * Auto-identify a configuration: net moment, circulation about the centroid,
 * checkerboard-staggered moment, and human-readable tags.
 */
export function describeState(theta, positions) {
    const n = theta.length;
    let mx = 0,
        my = 0,
        cx = 0,
        cy = 0;
    for (let i = 0; i < n; i++) {
        mx += Math.cos(theta[i]);
        my += Math.sin(theta[i]);
        cx += positions[i][0];
        cy += positions[i][1];
    }
    cx /= n;
    cy /= n;
    const netMoment = Math.hypot(mx, my) / n;
    let circ = 0,
        rsum = 0,
        sx = 0,
        sy = 0,
        onGrid = true;
    for (let i = 0; i < n; i++) {
        const x = positions[i][0],
            y = positions[i][1];
        const rx = x - cx,
            ry = y - cy;
        const cs = Math.cos(theta[i]),
            sn = Math.sin(theta[i]);
        circ += rx * sn - ry * cs;
        rsum += Math.hypot(rx, ry);
        const px = Math.round(x),
            py = Math.round(y);
        if (Math.abs(px - x) > 1e-9 || Math.abs(py - y) > 1e-9) onGrid = false;
        const s = (px + py) & 1 ? -1 : 1;
        sx += s * cs;
        sy += s * sn;
    }
    const circulation = rsum > 0 ? circ / rsum : 0;
    const staggered = onGrid ? Math.hypot(sx, sy) / n : 0;
    const tags = [];
    if (n > 1 && netMoment > 0.999) tags.push('ferromagnetic (all aligned)');
    else if (n > 1 && staggered > 0.999) tags.push('checkerboard antiferromagnetic');
    else if (netMoment < 1e-3) tags.push('zero net moment');
    else tags.push(`net moment ${netMoment.toFixed(2)}`);
    if (Math.abs(circulation) > 0.9) tags.push(`vortex (${circulation > 0 ? 'CCW' : 'CW'})`);
    return {netMoment, circulation, staggered, tags};
}

// ---- Basin records & summaries ---------------------------------------------

/** All perturbation records of a basin (single cores, normal modes, random rays). */
export function basinRecords(b) {
    return [...b.singles, ...b.modes, ...(b.randoms ?? [])];
}

function emptyBasin() {
    return {
        singles: [],
        modes: [],
        randoms: [],
        minSingleRadius: null,
        minModeRadius: null,
        meanRandomRadius: null,
        volumeFraction: null,
        nonMonotone: 0,
        targets: {},
    };
}

/** log Γ(x) (Lanczos approximation, |error| < 1e-13 for x > 0). */
function lnGamma(x) {
    const c = [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
    x -= 1;
    let a = c[0];
    const t = x + 7.5;
    for (let i = 1; i < 9; i++) a += c[i] / (x + i);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Rough basin volume as a fraction of configuration space [-π,π]^n, from the
 * escape radii r(Ω) measured along directions Ω uniform on the unit sphere:
 *     V = (1/n) ∫ r(Ω)^n dΩ = V_n(1) · ⟨r^n⟩,   fraction = V / (2π)^n.
 * The star-shaped-basin assumption and the radius cap make this an estimate
 * that is independent of the hit counts, not an exact measure. Clamped to ≤ 1.
 */
export function ballVolumeFraction(n, radii) {
    if (!radii.length || n <= 0) return null;
    const logs = radii.map((r) => n * Math.log(Math.max(r, 1e-300)));
    const mx = Math.max(...logs);
    let s = 0;
    for (const l of logs) s += Math.exp(l - mx);
    const logMean = mx + Math.log(s / radii.length);
    const logV = (n / 2) * Math.log(Math.PI) - lnGamma(n / 2 + 1) + logMean;
    return Math.min(1, Math.exp(logV - n * Math.log(2 * Math.PI)));
}

/** Recompute the derived summary fields of a basin from its perturbation records. */
export function summarizeBasin(b, n) {
    const minEsc = (recs) => {
        let m = null;
        for (const r of recs) if (r.escaped && (m === null || r.radius < m)) m = r.radius;
        return m;
    };
    if (!b.randoms) b.randoms = [];
    b.minSingleRadius = minEsc(b.singles);
    b.minModeRadius = minEsc(b.modes);
    if (b.randoms.length) {
        let s = 0;
        for (const r of b.randoms) s += r.radius;
        b.meanRandomRadius = s / b.randoms.length;
        b.volumeFraction = ballVolumeFraction(
            n,
            b.randoms.map((r) => r.radius),
        );
    } else {
        b.meanRandomRadius = null;
        b.volumeFraction = null;
    }
    b.nonMonotone = 0;
    b.targets = {};
    for (const r of basinRecords(b)) {
        if (r.monotone === false) b.nonMonotone++;
        if (r.escaped) b.targets[r.target] = (b.targets[r.target] || 0) + 1;
    }
    return b;
}

/** Re-point the catalog ids referenced by a basin record after the catalog was re-numbered. */
function remapBasinTargets(basin, idMap, n) {
    const f = (t) => (t && idMap.has(t) ? idMap.get(t) : t); // 0 / -1 / null are not ids
    for (const r of basinRecords(basin)) r.target = f(r.target);
    summarizeBasin(basin, n);
}

/**
 * Resolve escape records that landed on an uncatalogued minimum (target 0, with
 * the relaxed configuration kept in `landing`) against the current catalog —
 * used after a warm-started sweep has catalogued those minima. Returns the
 * number of records relinked.
 */
export function relinkBasinTargets(catalog, ops, tol = 1e-4) {
    let relinked = 0;
    for (const e of catalog) {
        if (!e.basin) continue;
        for (const r of basinRecords(e.basin)) {
            if (!r.escaped || r.target > 0 || !r.landing) continue;
            for (const t of catalog) {
                if (statesEquivalent(r.landing, t.theta, ops, tol)) {
                    r.target = t.id;
                    delete r.landing;
                    relinked++;
                    break;
                }
            }
        }
        summarizeBasin(e.basin, e.theta.length);
    }
    return relinked;
}

/**
 * Inter-basin connectivity: nodes are catalog entries, edges aggregate the
 * escape records from → to with their count and the smallest energy barrier
 * measured along any of the rays (null when no barrier was recorded).
 * Pseudo-targets: 0 = uncatalogued minimum ("new"), -1 = saddle / maximum.
 */
export function transitionGraph(catalog) {
    const nodes = catalog.map((e) => ({id: e.id, energy: e.energy, saddle: !!e.saddle, entry: e}));
    const edges = new Map();
    for (const e of catalog) {
        if (!e.basin) continue;
        for (const r of basinRecords(e.basin)) {
            if (!r.escaped || r.target === null || r.target === undefined) continue;
            const key = `${e.id}>${r.target}`;
            let ed = edges.get(key);
            if (!ed) {
                ed = {from: e.id, to: r.target, count: 0, barrier: null};
                edges.set(key, ed);
            }
            ed.count++;
            if (Number.isFinite(r.barrier) && (ed.barrier === null || r.barrier < ed.barrier))
                ed.barrier = r.barrier;
        }
    }
    return {nodes, edges: [...edges.values()]};
}

// ---- Stable-state sweep -----------------------------------------------------

/**
 * Exhaustive-ish search for stable equilibria: relax from structured seeds and
 * many random initial conditions, and merge configurations that are
 * equivalent under the lattice symmetries + global flip. Every converged
 * stationary point is kept — saddle points (negative Hessian curvature) are
 * common in dipole lattices and are tagged rather than discarded. Work is
 * done incrementally via runFor() so a UI can keep animating.
 *
 * Structured seeds and uniformly random starts are bookkept separately:
 * `entry.randomHits / sweep.randomDone` is an unbiased estimate of the
 * fraction of configuration space (the torus [-π,π]^n, uniform measure) that
 * relaxes into a state — the structured seeds would bias it.
 *
 * Pass `opts.catalog` (+ `opts.randomStarts`) to extend an existing catalog
 * instead of starting from scratch; the structured seeds are then skipped.
 * Pass `opts.seeds` (configurations) to warm-start from explicit points —
 * e.g. the boundary landings found by BasinAnalysis — before random starts.
 */
export class MinimaSweep {
    constructor(positions, pairs, n, opts = {}) {
        this.positions = positions;
        this.pairs = pairs;
        this.n = n;
        this.matchTol = opts.matchTol ?? 1e-4;
        this.relaxOpts = {maxIter: opts.maxIter ?? 800, tol: opts.tol ?? 1e-8};
        this.random = opts.random ?? Math.random;
        this.ops = latticeSymmetries(positions);
        /** Catalogue of distinct states, sorted by energy (shared with a seeding catalog). */
        this.catalog = opts.catalog ? opts.catalog.slice() : [];
        for (const e of this.catalog) {
            if (!Number.isFinite(e.randomHits)) e.randomHits = 0;
            if (e.basin === undefined) e.basin = null;
        }
        /** Uniformly random starts run so far (structured seeds excluded). */
        this.randomDone = opts.randomStarts ?? 0;
        this.done = 0;
        this.unconverged = 0;
        this.saddles = 0;
        if (opts.seeds) {
            this._seeds = opts.seeds.map((s) => Float64Array.from(s));
            // tiny jitter so seeds sitting exactly on a saddle can escape
            for (const s of this._seeds)
                for (let i = 0; i < n; i++) s[i] += 1e-2 * (this.random() - 0.5);
        } else {
            this._seeds = this.catalog.length ? [] : this._structuredSeeds();
        }
        this.seedCount = this._seeds.length;
        this.total = Math.max(1, opts.starts ?? 200, this.seedCount);
    }
    _structuredSeeds() {
        const {n, positions} = this;
        const seeds = [];
        for (const a of [0, Math.PI / 2, Math.PI / 4, -Math.PI / 4])
            seeds.push(new Float64Array(n).fill(a));
        const checker = new Float64Array(n),
            rows = new Float64Array(n),
            cols = new Float64Array(n);
        for (let i = 0; i < n; i++) {
            const px = Math.round(positions[i][0]),
                py = Math.round(positions[i][1]);
            checker[i] = (px + py) & 1 ? Math.PI : 0;
            rows[i] = py & 1 ? Math.PI : 0;
            cols[i] = px & 1 ? Math.PI / 2 : -Math.PI / 2;
        }
        seeds.push(checker, rows, cols);
        // tiny jitter so seeds sitting exactly on a saddle can escape
        for (const s of seeds) for (let i = 0; i < n; i++) s[i] += 1e-2 * (this.random() - 0.5);
        return seeds;
    }
    get finished() {
        return this.done >= this.total;
    }
    /** Run one relaxation. Returns the catalog entry it landed in (or null). */
    runOne() {
        const {n, pairs} = this;
        let theta0;
        let isRandom = false;
        if (this._seeds.length) theta0 = this._seeds.shift();
        else {
            isRandom = true;
            theta0 = new Float64Array(n);
            for (let i = 0; i < n; i++) theta0[i] = (this.random() * 2 - 1) * Math.PI;
        }
        const r = relax(theta0, pairs, n, this.relaxOpts);
        this.done++;
        if (isRandom) this.randomDone++;
        if (!(r.gradNorm < 1e-6)) {
            this.unconverged++;
            return null;
        }
        const entry = this.addState(r.theta);
        if (isRandom && entry) entry.randomHits++;
        return entry;
    }
    /** Run relaxations until finished or `budgetMs` elapsed. Returns `finished`. */
    runFor(budgetMs = 20) {
        if (this.finished) return true;
        const t0 = now();
        do {
            this.runOne();
        } while (!this.finished && now() - t0 < budgetMs);
        return this.finished;
    }
    /**
     * Insert a converged stationary point. States equivalent to an existing
     * entry only bump its hit counter. Saddle points (negative Hessian
     * curvature) are kept and tagged rather than discarded — for these dipole
     * lattices they are a normal, expected outcome. Ids follow energy order;
     * basin records referring to ids are re-pointed when the order changes.
     */
    addState(theta) {
        const {n, pairs} = this;
        if (pairs.length === 0 && this.catalog.length) {
            this.catalog[0].hits++;
            return this.catalog[0];
        }
        for (const entry of this.catalog) {
            if (statesEquivalent(theta, entry.theta, this.ops, this.matchTol)) {
                entry.hits++;
                return entry;
            }
        }
        const H = hessian(theta, pairs, n);
        const lambdaMin = jacobiEigen(H).values[0];
        const isSaddle = lambdaMin < -1e-7;
        if (isSaddle) this.saddles++;
        let stabilizer = 0;
        for (const img of symmetryImages(theta, this.ops)) {
            if (maxAngleDiff(img, theta) < this.matchTol) stabilizer++;
        }
        const groupSize = 2 * this.ops.length;
        const entry = {
            id: 0,
            theta: Float64Array.from(theta),
            energy: energy(theta, pairs),
            lambdaMin,
            soft: !isSaddle && lambdaMin < 1e-6,
            saddle: isSaddle,
            hits: 1,
            randomHits: 0,
            stabilizer,
            orbit: Math.max(1, Math.round(groupSize / Math.max(1, stabilizer))),
            basin: null,
            ...describeState(theta, this.positions),
        };
        if (entry.soft) entry.tags.push('soft mode (marginal)');
        if (entry.saddle) entry.tags.push('saddle point');
        this.catalog.push(entry);
        this._reindex();
        return entry;
    }
    /** Sort by energy, assign ids 1..N and keep basin target references consistent. */
    _reindex() {
        const oldIds = this.catalog.map((e) => [e, e.id]);
        this.catalog.sort((a, b) => a.energy - b.energy);
        this.catalog.forEach((e, i) => (e.id = i + 1));
        const idMap = new Map();
        for (const [e, old] of oldIds) if (old > 0) idMap.set(old, e.id);
        let changed = false;
        for (const [old, nu] of idMap) if (old !== nu) changed = true;
        if (!changed) return;
        for (const e of this.catalog) if (e.basin) remapBasinTargets(e.basin, idMap, this.n);
    }
}

// ---- Basins of attraction ---------------------------------------------------
//
// Complementary estimates for every stable catalog entry θ*:
//   1. Volume share — the fraction of *uniformly random* sweep starts that
//      relaxed into the state (structured seeds are excluded), see
//      basinFraction(). Standard binomial error bar.
//   2. Boundary distance — for each core i and sign ±, for each normal mode
//      v_k (unit eigenvector of H(θ*)) and sign ±, and for a set of random
//      directions u (uniform on the unit sphere in R^n), bisect on the
//      amplitude δ of the perturbation
//          θ* ± δ e_i     (single core)     δ ∈ (0, π]
//          θ* ± δ v_k     (normal mode)     δ ∈ (0, π√n]
//          θ* + δ u       (random ray)      δ ∈ (0, π√n]
//      for the smallest δ whose relaxation no longer returns to θ* itself.
//      The relaxed configuration just beyond the boundary is matched against
//      the catalog, which maps the connectivity between basins.
//   3. Barrier — the energy climbed along the ray up to the boundary
//      crossing, max_{0≤s≤δ} U(θ* + s·dir) − U(θ*): an upper bound on the
//      true saddle height between the two basins along this particular path.
//   4. Shape — the random rays give a direction-averaged escape radius and a
//      rough basin volume (ballVolumeFraction) independent of the hit counts.
//
// Bisection assumes "returns to θ*" is monotone in δ along the ray. After the
// bracket has converged a few extra amplitudes are probed (below the last
// returning amplitude, expected to return; beyond the boundary, expected to
// escape); a ray whose spot-checks disagree is flagged `monotone: false` and
// its radius is the smallest escaping amplitude actually observed.
//
// Results are stored on entry.basin:
//   { singles: [{i, sign, radius, escaped, target, barrier, monotone}],
//     modes:   [{k, lambda, sign, radius, escaped, target, barrier, monotone}],
//     randoms: [{r, dir, sign, radius, escaped, target, barrier, monotone}],
//     minSingleRadius, minModeRadius, meanRandomRadius, volumeFraction,
//     nonMonotone, targets: {targetId: count} }
// target: catalog id (> 0); 0 = converged to an uncatalogued minimum (the
// relaxed configuration is kept in `landing` so a warm-started sweep can
// catalogue it and relinkBasinTargets() can resolve the id); -1 = landed on
// a saddle / maximum; null = never escaped within the range.

/** Basin volume share estimated from uniformly random starts, or null if there were none. */
export function basinFraction(entry, randomStarts) {
    if (!(randomStarts > 0)) return null;
    const p = (entry.randomHits ?? 0) / randomStarts;
    return {p, err: Math.sqrt((p * (1 - p)) / randomStarts), n: randomStarts};
}

export class BasinAnalysis {
    /**
     * @param positions  magnet positions (cell units) — for the symmetry group
     * @param pairs      buildPairs() output
     * @param n          magnet count
     * @param catalog    MinimaSweep catalog (entries are annotated in place)
     * @param opts       { singles, modes, randomDirs, spotChecks, bisectTol, barrierSamples, random }
     */
    constructor(positions, pairs, n, catalog, opts = {}) {
        this.pairs = pairs;
        this.n = n;
        this.catalog = catalog;
        this.ops = latticeSymmetries(positions);
        this.matchTol = opts.matchTol ?? 1e-4;
        /** Bisection stops when the bracket is narrower than this (radians of amplitude δ). */
        this.bisectTol = opts.bisectTol ?? 5e-3;
        /** Extra probes per ray used to spot-check monotonicity. */
        this.spotChecks = Math.max(0, opts.spotChecks ?? 4);
        /** Energy samples along a ray for the barrier estimate. */
        this.barrierSamples = opts.barrierSamples ?? 32;
        this.random = opts.random ?? Math.random;
        this.relaxOpts = {maxIter: opts.maxIter ?? 800, tol: opts.tol ?? 1e-8};
        const doSingles = opts.singles ?? true;
        const doModes = opts.modes ?? true;
        const randomDirs = Math.max(0, opts.randomDirs ?? 8);
        /** Relaxed landings on uncatalogued stationary points — warm-start seeds for a sweep. */
        this.seeds = [];
        /** Number of escape records that reached an uncatalogued *minimum*. */
        this.newCount = 0;
        this.tasks = [];
        const hiMax = Math.PI * Math.sqrt(n);
        for (const entry of catalog) {
            if (entry.saddle) {
                entry.basin = null; // a saddle has a measure-zero basin
                continue;
            }
            entry.basin = emptyBasin();
            if (doSingles) {
                for (let i = 0; i < n; i++) {
                    for (const sign of [1, -1]) {
                        this.tasks.push(
                            this._task(entry, {kind: 'single', i, sign, dir: null, hiMax: Math.PI}),
                        );
                    }
                }
            }
            if (doModes) {
                const {values, vectors} = jacobiEigen(hessian(entry.theta, pairs, n));
                for (let k = 0; k < n; k++) {
                    for (const sign of [1, -1]) {
                        this.tasks.push(
                            this._task(entry, {
                                kind: 'mode',
                                k,
                                lambda: values[k],
                                sign,
                                dir: vectors[k],
                                hiMax,
                            }),
                        );
                    }
                }
            }
            for (let r = 0; r < randomDirs; r++) {
                this.tasks.push(
                    this._task(entry, {
                        kind: 'random',
                        r,
                        sign: 1,
                        dir: this._randomDirection(),
                        hiMax,
                    }),
                );
            }
        }
        this.total = this.tasks.length;
        this.done = 0;
        this._cursor = 0;
    }
    /** Unit vector uniform on the sphere S^{n-1} (normalised Gaussian). */
    _randomDirection() {
        const n = this.n;
        const v = new Float64Array(n);
        let s = 0;
        do {
            for (let i = 0; i < n; i++) {
                const u1 = Math.max(this.random(), 1e-300),
                    u2 = this.random();
                v[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
            }
            s = norm(v);
        } while (s < 1e-9);
        for (let i = 0; i < n; i++) v[i] /= s;
        return v;
    }
    _task(entry, spec) {
        return {
            entry,
            ...spec,
            phase: 'bisect',
            lo: 0,
            hi: null,
            probe: spec.hiMax,
            minEscape: null, // smallest amplitude observed to escape
            target: null, // target at minEscape
            landing: null, // relaxed configuration at minEscape when target ≤ 0
            altTarget: null, // first catalogued target seen anywhere along the ray
            monotone: true,
            checks: null,
        };
    }
    get finished() {
        return this.done >= this.total;
    }
    /** Displace entry.theta by amplitude `amp` along the task's ray. */
    _displaced(task, amp) {
        const theta = Float64Array.from(task.entry.theta);
        const d = task.sign * amp;
        if (task.kind === 'single') theta[task.i] += d;
        else for (let i = 0; i < this.n; i++) theta[i] += d * task.dir[i];
        return theta;
    }
    /** Perturb entry.theta by the task's current amplitude, relax, and classify the outcome. */
    _probe(task) {
        const {entry} = task;
        const r = relax(this._displaced(task, task.probe), this.pairs, this.n, this.relaxOpts);
        const converged = r.gradNorm < 1e-6;
        const stays = converged && maxAngleDiff(r.theta, entry.theta) < this.matchTol;
        let target = 0,
            landing = null;
        if (converged && !stays) {
            for (const e of this.catalog) {
                if (statesEquivalent(r.theta, e.theta, this.ops, this.matchTol)) {
                    target = e.id;
                    break;
                }
            }
            if (target === 0) {
                // uncatalogued stationary point: minimum (0) or saddle/maximum (-1)?
                const lmin = jacobiEigen(hessian(r.theta, this.pairs, this.n)).values[0];
                if (lmin < -1e-7) target = -1;
                landing = r.theta;
            }
        }
        return {stays, target, landing};
    }
    _noteEscape(task, amp, target, landing) {
        if (task.minEscape === null || amp < task.minEscape) {
            task.minEscape = amp;
            task.target = target;
            task.landing = landing;
        }
        if (target > 0 && task.altTarget === null) task.altTarget = target;
    }
    /** Energy climbed along the ray up to amplitude `delta` (upper bound on the barrier). */
    _barrier(task, delta) {
        const K = Math.max(1, this.barrierSamples);
        let maxU = task.entry.energy;
        for (let s = 1; s <= K; s++) {
            maxU = Math.max(maxU, energy(this._displaced(task, (delta * s) / K), this.pairs));
        }
        return maxU - task.entry.energy;
    }
    /** Schedule the monotonicity spot-checks once the bracket has converged. */
    _beginChecks(task) {
        const m = this.spotChecks;
        const checks = [];
        if (task.hi === null) {
            // never escaped within range: sample the interior, all expected to return
            for (let k = 1; k <= m; k++)
                checks.push({a: (task.hiMax * k) / (m + 1), expectStay: true});
        } else {
            const below = Math.floor(m / 2),
                above = m - below;
            for (let k = 1; k <= below; k++)
                checks.push({a: (task.lo * k) / (below + 1), expectStay: true});
            for (let k = 1; k <= above; k++) {
                checks.push({
                    a: task.hi + ((task.hiMax - task.hi) * k) / (above + 1),
                    expectStay: false,
                });
            }
        }
        task.checks = checks.filter((c) => c.a > 1e-9 && c.a <= task.hiMax + 1e-12);
        task.phase = 'check';
        if (!task.checks.length) return this._finish(task);
        task.probe = task.checks[0].a;
        return null;
    }
    /** One relaxation of the current ray. Returns the finished record, if any. */
    runOne() {
        if (this.finished) return null;
        const task = this.tasks[this._cursor];
        const amp = task.probe;
        const {stays, target, landing} = this._probe(task);
        if (task.phase === 'check') {
            const c = task.checks.shift();
            if (stays !== c.expectStay) task.monotone = false;
            if (!stays) this._noteEscape(task, amp, target, landing);
            if (!task.checks.length) return this._finish(task);
            task.probe = task.checks[0].a;
            return null;
        }
        if (task.hi === null) {
            // first probe at the largest amplitude
            if (stays) return this._beginChecks(task);
            task.hi = amp;
            this._noteEscape(task, amp, target, landing);
        } else if (stays) {
            task.lo = amp;
        } else {
            task.hi = amp;
            this._noteEscape(task, amp, target, landing);
        }
        if (task.hi - task.lo <= this.bisectTol) return this._beginChecks(task);
        task.probe = 0.5 * (task.lo + task.hi);
        return null;
    }
    _finish(task) {
        const b = task.entry.basin;
        const escaped = task.minEscape !== null;
        const radius = escaped ? task.minEscape : task.hiMax;
        let target = null;
        if (escaped) {
            target = task.target;
            // a saddle landing (or an unconverged one) is replaced by the first catalogued
            // target seen further along the ray; a genuinely new minimum keeps 0 + landing
            if ((target === -1 || (target === 0 && !task.landing)) && task.altTarget !== null) {
                target = task.altTarget;
            }
        }
        const barrier = escaped ? this._barrier(task, radius) : null;
        const rec = {sign: task.sign, radius, escaped, target, barrier, monotone: task.monotone};
        if (task.kind === 'single') {
            rec.i = task.i;
            b.singles.push(rec);
        } else if (task.kind === 'mode') {
            rec.k = task.k;
            rec.lambda = task.lambda;
            b.modes.push(rec);
        } else {
            rec.r = task.r;
            rec.dir = Float64Array.from(task.dir);
            b.randoms.push(rec);
        }
        if (escaped && target <= 0 && task.landing) {
            rec.landing = task.landing;
            this.seeds.push(task.landing);
            if (target === 0) this.newCount++;
        }
        summarizeBasin(b, this.n);
        this.done++;
        this._cursor++;
        return rec;
    }
    /** Run probes until finished or `budgetMs` elapsed. Returns `finished`. */
    runFor(budgetMs = 20) {
        if (this.finished) return true;
        const t0 = now();
        do {
            this.runOne();
        } while (!this.finished && now() - t0 < budgetMs);
        return this.finished;
    }
}
