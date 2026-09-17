import {describe, it, assert, assertClose} from './harness.js';
import {buildPairs, gradient} from '../src/model/physics.js';
import {
    relax,
    analyze,
    latticeSymmetries,
    statesEquivalent,
    MinimaSweep,
    BasinAnalysis,
    basinFraction,
    ballVolumeFraction,
    transitionGraph,
    relinkBasinTargets,
    summarizeBasin,
} from '../src/model/analysis.js';
import {jacobiEigen, luSolve} from '../src/math/linalg.js';

const params = {k: 1.0, m: 1.0, I: 1.0};
const pos = [
    [0, 0],
    [1, 0],
];

/** Small deterministic PRNG so the random-direction tests are reproducible. */
function lcg(seed) {
    let s = seed >>> 0;
    return () => {
        s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
        return (s + 0.5) / 4294967296;
    };
}

describe('linalg: luSolve', () => {
    it('solves a 3x3 system', () => {
        const A = [
            new Float64Array([2, 1, 1]),
            new Float64Array([1, 3, 2]),
            new Float64Array([1, 0, 0]),
        ];
        const b = new Float64Array([4, 5, 6]);
        const x = luSolve(A, b);
        for (let i = 0; i < 3; i++) {
            let s = 0;
            for (let j = 0; j < 3; j++) s += A[i][j] * x[j];
            assertClose(s, b[i], 1e-10, `row ${i}`);
        }
    });

    it('throws on singular matrix', () => {
        const A = [new Float64Array([1, 2]), new Float64Array([2, 4])];
        let threw = false;
        try {
            luSolve(A, new Float64Array([1, 1]));
        } catch {
            threw = true;
        }
        assert(threw, 'expected singular-matrix error');
    });
});

describe('linalg: jacobiEigen', () => {
    it('diagonal matrix eigenvalues', () => {
        const A = [
            new Float64Array([3, 0, 0]),
            new Float64Array([0, 1, 0]),
            new Float64Array([0, 0, 2]),
        ];
        const {values} = jacobiEigen(A);
        assertClose(values[0], 1, 1e-10);
        assertClose(values[1], 2, 1e-10);
        assertClose(values[2], 3, 1e-10);
    });

    it('known 2x2 symmetric matrix', () => {
        const A = [new Float64Array([2, 1]), new Float64Array([1, 2])];
        const {values, vectors} = jacobiEigen(A);
        assertClose(values[0], 1, 1e-10);
        assertClose(values[1], 3, 1e-10);
        for (let k = 0; k < 2; k++) {
            const v = vectors[k];
            for (let i = 0; i < 2; i++) {
                let s = 0;
                for (let j = 0; j < 2; j++) s += A[i][j] * v[j];
                assertClose(s, values[k] * v[i], 1e-9, `Av=λv k=${k} i=${i}`);
            }
        }
    });
});

describe('analysis: 2-magnet equilibrium & modes', () => {
    it('relax finds the aligned ground state to near machine precision', () => {
        const pairs = buildPairs(pos, params);
        const {theta, gradNorm} = relax(new Float64Array([0.5, -0.3]), pairs, 2);
        assert(gradNorm < 1e-9, `gradNorm ${gradNorm}`);
        const g = gradient(theta, pairs, 2);
        assertClose(g[0], 0, 1e-9);
        assertClose(g[1], 0, 1e-9);
        // both along ±x, and parallel to each other
        assertClose(Math.abs(Math.sin(theta[0])), 0, 1e-8, 'θ0 on x-axis');
        assertClose(Math.abs(Math.sin(theta[1])), 0, 1e-8, 'θ1 on x-axis');
        assertClose(Math.cos(theta[0] - theta[1]), 1, 1e-8, 'parallel');
    });

    it('normal modes match closed form ω = 1 (anti-symmetric) and √3 (symmetric)', () => {
        const pairs = buildPairs(pos, params);
        const {theta} = relax(new Float64Array([0.1, -0.1]), pairs, 2);
        const res = analyze(theta, pairs, 2, params);
        assertClose(res.omega[0], 1, 1e-6, 'ω0');
        assertClose(res.omega[1], Math.sqrt(3), 1e-6, 'ω1');
        // mode shapes: eigenvalue 1 -> (1,-1)/√2 ; eigenvalue 3 -> (1,1)/√2
        const v0 = res.modes[0],
            v1 = res.modes[1];
        assertClose(Math.abs(v0[0] * v0[1] * 2), 1, 1e-6, 'mode0 anti-symmetric');
        assert(v0[0] * v0[1] < 0, 'mode0 components have opposite sign');
        assert(v1[0] * v1[1] > 0, 'mode1 components have same sign');
        // normalized coupling
        assertClose(res.C[0][0], 1, 1e-12);
        assertClose(res.C[0][1], 0.5, 1e-9);
    });

    it('flags the anti-aligned (head-to-head) configuration as unstable', () => {
        const pairs = buildPairs(pos, params);
        const res = analyze(new Float64Array([0, Math.PI]), pairs, 2, params);
        assert(res.values[0] < 0, `expected a negative eigenvalue, got ${res.values[0]}`);
        assert(res.omega[0] < 0, 'omega sign should flag instability');
    });
});

describe('analysis: lattice symmetries & stable-state sweep', () => {
    it('a 2-magnet row keeps the 4 D4 operations that preserve it', () => {
        const ops = latticeSymmetries(pos);
        const names = ops.map((o) => o.name).sort();
        assert(names.join(',') === 'e,mx,my,r180', `got ${names}`);
        const r180 = ops.find((o) => o.name === 'r180');
        assert(r180.perm[0] === 1 && r180.perm[1] === 0, 'r180 swaps the two magnets');
    });
    it('a 2x2 square keeps all 8 operations', () => {
        const sq = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ];
        assert(latticeSymmetries(sq).length === 8);
    });
    it('treats flipped / mirrored / rotated states as equivalent', () => {
        const ops = latticeSymmetries(pos);
        const eq = (a, b) => statesEquivalent(Float64Array.from(a), Float64Array.from(b), ops);
        assert(eq([0, 0], [Math.PI, Math.PI]), 'global flip');
        assert(eq([0.3, -0.3], [-0.3, 0.3]), 'mirror across the row axis');
        assert(eq([0.3, 0.7], [Math.PI + 0.7, Math.PI + 0.3]), 'rotation by 180° swaps & rotates');
        assert(!eq([0, 0], [0, Math.PI]), 'head-to-head is a different state');
    });
    it('sweep catalogs exactly one stable state for two magnets (U = -2)', () => {
        const pairs = buildPairs(pos, params);
        const sw = new MinimaSweep(pos, pairs, 2, {starts: 40});
        while (!sw.runFor(1000));
        assert(sw.done === 40);
        assert(sw.catalog.length === 1, `expected 1 state, got ${sw.catalog.length}`);
        const e = sw.catalog[0];
        assertClose(e.energy, -2, 1e-8, 'ground-state energy');
        assert(e.lambdaMin > 0.5, `should be stiff, λmin=${e.lambdaMin}`);
        assert(e.tags.includes('ferromagnetic (all aligned)'), `tags ${e.tags}`);
        assert(e.orbit === 2, `orbit {(0,0),(π,π)} expected 2, got ${e.orbit}`);
        assert(e.hits + sw.saddles + sw.unconverged === 40, 'every start is accounted for');
    });
    it('sweep entries are stable and pairwise non-equivalent (2x2 square)', () => {
        const sq = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ];
        const pairs = buildPairs(sq, params);
        const sw = new MinimaSweep(sq, pairs, 4, {starts: 60});
        while (!sw.runFor(1000));
        const cat = sw.catalog;
        assert(cat.length >= 1, 'found at least one minimum');
        for (let a = 0; a < cat.length; a++) {
            assert(cat[a].lambdaMin >= -1e-7, `entry ${a} not stable`);
            assert(cat[a].id === a + 1, 'ids follow energy order');
            if (a > 0) assert(cat[a].energy >= cat[a - 1].energy, 'sorted by energy');
            for (let b = a + 1; b < cat.length; b++) {
                assert(!statesEquivalent(cat[a].theta, cat[b].theta, sw.ops), `entries ${a},${b} duplicate`);
            }
        }
    });
    it('explicit seeds warm-start a sweep and are not counted as random starts', () => {
        const pairs = buildPairs(pos, params);
        const sw = new MinimaSweep(pos, pairs, 2, {starts: 1, seeds: [[0.3, 0.2]]});
        assert(sw.seedCount === 1 && sw.total === 1);
        while (!sw.runFor(1000));
        assert(sw.done === 1 && sw.randomDone === 0, 'seed run is not a random start');
        assert(sw.catalog.length === 1);
        assertClose(sw.catalog[0].energy, -2, 1e-8);
        // seeds extend an existing catalog without re-running the structured seeds
        const sw2 = new MinimaSweep(pos, pairs, 2, {
            starts: 2,
            catalog: sw.catalog,
            randomStarts: 0,
            seeds: [[Math.PI, Math.PI], [0.1, -0.1]],
        });
        while (!sw2.runFor(1000));
        assert(sw2.done === 2 && sw2.randomDone === 0 && sw2.catalog.length === 1);
        assert(sw2.catalog[0].hits === 3, `hits accumulate, got ${sw2.catalog[0].hits}`);
    });
});

describe('analysis: basins of attraction', () => {
    it('sweep bookkeeps uniformly random starts apart from structured seeds and can be extended', () => {
        const pairs = buildPairs(pos, params);
        const sw = new MinimaSweep(pos, pairs, 2, {starts: 40});
        while (!sw.runFor(1000));
        assert(sw.randomDone === 40 - 7, `7 structured seeds + 33 random, got ${sw.randomDone} random`);
        assert(sw.catalog.length === 1);
        const e = sw.catalog[0];
        assert(e.randomHits <= sw.randomDone && e.randomHits <= e.hits, 'random hits ⊆ hits');
        const bf = basinFraction(e, sw.randomDone);
        assert(bf.p >= 0 && bf.p <= 1 && bf.err >= 0, 'fraction with error bar');
        assert(basinFraction(e, 0) === null, 'no estimate without random starts');
        const hitsBefore = e.hits;
        const sw2 = new MinimaSweep(pos, pairs, 2, {starts: 10, catalog: sw.catalog, randomStarts: sw.randomDone});
        while (!sw2.runFor(1000));
        assert(sw2.catalog.length === 1 && sw2.catalog[0] === e, 'seeded sweep extends the same entries');
        assert(sw2.randomDone === sw.randomDone + 10, 'seeded sweep skips the structured seeds');
        assert(e.hits + sw2.unconverged === hitsBefore + 10, 'hits keep accumulating');
    });
    it('bisection along both 2-magnet normal modes finds the saddle at δ = π/√2 with barriers 1 and 3', () => {
        // Along v=(1,1)/√2 the state is (a,a) with U = 1 - 3cos²a; along (1,-1)/√2 it is (a,-a)
        // with U = cos2a - 3cos²a. Both leave the (0,0) basin for the flipped image (π,π) at
        // a = π/2, i.e. amplitude δ = a√2 = π/√2. The energy climbed up to the crossing is
        // U(π/2) - U(0): 1 - (-2) = 3 for the symmetric mode and -1 - (-2) = 1 for the other.
        const pairs = buildPairs(pos, params);
        const sw = new MinimaSweep(pos, pairs, 2, {starts: 8});
        while (!sw.runFor(1000));
        assert(sw.catalog.length === 1);
        const ba = new BasinAnalysis(pos, pairs, 2, sw.catalog, {bisectTol: 1e-3, randomDirs: 0});
        assert(ba.total === 8, `2 cores × 2 signs + 2 modes × 2 signs, got ${ba.total}`);
        while (!ba.runFor(1000));
        assert(ba.done === 8);
        const b = sw.catalog[0].basin;
        assert(b && b.modes.length === 4 && b.singles.length === 4, 'all perturbations recorded');
        for (const r of b.modes) {
            assert(r.escaped, `mode ${r.k} sign ${r.sign} should escape`);
            assertClose(r.radius, Math.PI / Math.SQRT2, 2e-3, `mode ${r.k} sign ${r.sign} radius`);
            assert(r.target === 1, `lands on the flipped image of the same state, got ${r.target}`);
            assert(r.monotone === true, `mode ${r.k} sign ${r.sign} should be monotone`);
            assertClose(r.barrier, r.k === 0 ? 1 : 3, 5e-3, `mode ${r.k} barrier`);
        }
        assertClose(b.modes.find((r) => r.k === 0).lambda, 1, 1e-6, 'λ0');
        assertClose(b.modes.find((r) => r.k === 1).lambda, 3, 1e-6, 'λ1');
        assertClose(b.minModeRadius, Math.PI / Math.SQRT2, 2e-3, 'min mode radius');
        assert(b.targets['1'] >= 4, `connectivity counts the 4 mode escapes, got ${JSON.stringify(b.targets)}`);
        assert(b.randoms.length === 0 && b.meanRandomRadius === null && b.volumeFraction === null);
        for (const r of b.singles) {
            assert(r.radius > 0 && r.radius <= Math.PI + 1e-12, `single radius ${r.radius}`);
            assert(!r.escaped || [1, 0, -1].includes(r.target), `single target ${r.target}`);
            assert(typeof r.monotone === 'boolean');
            assert(!r.escaped || (r.barrier >= 0 && r.barrier <= 4 + 1e-9), `single barrier ${r.barrier}`);
        }
        // transition graph: one node, a self-edge onto the flipped image, lowest barrier along a ray = 1
        const g = transitionGraph(sw.catalog);
        assert(g.nodes.length === 1 && g.nodes[0].id === 1);
        const self = g.edges.find((e) => e.from === 1 && e.to === 1);
        assert(self && self.count >= 4, `self-transition edge, got ${JSON.stringify(g.edges)}`);
        assert(self.barrier >= 0 && self.barrier <= 1 + 5e-3, `min barrier ${self.barrier}`);
    });
    it('random directions give a direction-averaged radius and a volume estimate', () => {
        const pairs = buildPairs(pos, params);
        const sw = new MinimaSweep(pos, pairs, 2, {starts: 8});
        while (!sw.runFor(1000));
        const ba = new BasinAnalysis(pos, pairs, 2, sw.catalog, {
            singles: false,
            modes: false,
            randomDirs: 6,
            spotChecks: 2,
            random: lcg(42),
        });
        assert(ba.total === 6, `6 random rays, got ${ba.total}`);
        while (!ba.runFor(1000));
        const b = sw.catalog[0].basin;
        assert(b.randoms.length === 6);
        for (const r of b.randoms) {
            assert(r.dir.length === 2, 'direction stored');
            assertClose(Math.hypot(r.dir[0], r.dir[1]), 1, 1e-12, 'unit direction');
            assert(r.radius > 0 && r.radius <= Math.PI * Math.SQRT2 + 1e-12, `radius ${r.radius}`);
            assert(typeof r.escaped === 'boolean' && typeof r.monotone === 'boolean');
        }
        assert(b.meanRandomRadius > 0, 'mean radius');
        assert(b.volumeFraction > 0 && b.volumeFraction <= 1, `volume fraction ${b.volumeFraction}`);
        // closed forms: a disc of radius π covers π/4 of the 2-torus; a segment of half-length π covers the circle
        assertClose(ballVolumeFraction(2, [Math.PI, Math.PI, Math.PI]), Math.PI / 4, 1e-12, 'disc');
        assertClose(ballVolumeFraction(1, [Math.PI]), 1, 1e-12, 'segment');
        assert(ballVolumeFraction(3, []) === null);
    });
    it('saddle entries get no basin record', () => {
        const pairs = buildPairs(pos, params);
        const fake = [
            {id: 1, theta: new Float64Array([0, 0]), energy: -2, lambdaMin: 1, saddle: false, basin: null},
            {id: 2, theta: new Float64Array([0, Math.PI]), energy: 2, lambdaMin: -3, saddle: true, basin: null},
        ];
        const ba = new BasinAnalysis(pos, pairs, 2, fake, {randomDirs: 0});
        assert(ba.total === 8, 'only the minimum is analysed');
        assert(fake[0].basin !== null && fake[1].basin === null);
    });
    it('relinkBasinTargets resolves "new" landings once the state is catalogued', () => {
        const ops = latticeSymmetries(pos);
        const fake = [
            {
                id: 1,
                theta: new Float64Array([0, 0]),
                energy: -2,
                saddle: false,
                basin: {
                    singles: [
                        {i: 0, sign: 1, radius: 1, escaped: true, target: 0, barrier: 1, monotone: true, landing: new Float64Array([Math.PI, Math.PI])},
                        {i: 1, sign: 1, radius: 2, escaped: true, target: 0, barrier: 1, monotone: false, landing: new Float64Array([0.5, 0.5])},
                    ],
                    modes: [],
                    randoms: [],
                },
            },
        ];
        summarizeBasin(fake[0].basin, 2);
        assert(fake[0].basin.targets['0'] === 2 && fake[0].basin.nonMonotone === 1);
        const n = relinkBasinTargets(fake, ops);
        assert(n === 1, `one landing matches the catalog, got ${n}`);
        const [a, b] = fake[0].basin.singles;
        assert(a.target === 1 && a.landing === undefined, 'relinked record drops its landing');
        assert(b.target === 0 && b.landing, 'unmatched landing is kept for later');
        assert(fake[0].basin.targets['1'] === 1 && fake[0].basin.targets['0'] === 1);
        const g = transitionGraph(fake);
        assert(g.edges.length === 2 && g.edges.every((e) => e.count === 1 && e.barrier === 1));
    });
});