// Versioned JSON import/export with schema validation.
//
// The document optionally embeds the stable-state catalog (with basin data)
// produced by the Analyze sweep, so an arrangement and its analysis travel
// together. The field is optional and ignored by older readers.

const VERSION = 1;

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const nonNegInt = (v, dflt) => (Number.isInteger(v) && v >= 0 ? v : dflt);
/** Common fields of a basin perturbation record (the transient `landing` is not persisted). */
function recToJSON(r) {
     return {
         sign: r.sign,
         radius: r.radius,
         escaped: r.escaped,
         target: r.target,
         barrier: Number.isFinite(r.barrier) ? r.barrier : null,
         monotone: r.monotone !== false,
     };
}


function basinToJSON(b) {
    return {
         singles: b.singles.map((r) => ({i: r.i, ...recToJSON(r)})),
         modes: b.modes.map((r) => ({k: r.k, lambda: r.lambda ?? null, ...recToJSON(r)})),
         randoms: (b.randoms ?? []).map((r) => ({
             r: r.r,
             dir: r.dir ? Array.from(r.dir) : null,
             ...recToJSON(r),
         })),
        minSingleRadius: b.minSingleRadius,
        minModeRadius: b.minModeRadius,
         meanRandomRadius: b.meanRandomRadius ?? null,
         volumeFraction: b.volumeFraction ?? null,
         nonMonotone: b.nonMonotone ?? 0,
        targets: {...b.targets},
    };
}

/** Convert a catalog (array of MinimaSweep entries) to plain JSON-able data. */
export function catalogToJSON(entries, randomStarts = 0) {
    return {
        randomStarts,
        entries: entries.map((e) => ({
            id: e.id,
            theta: Array.from(e.theta),
            energy: e.energy,
            lambdaMin: e.lambdaMin,
            hits: e.hits,
            randomHits: e.randomHits ?? 0,
            orbit: e.orbit,
            stabilizer: e.stabilizer,
            saddle: !!e.saddle,
            soft: !!e.soft,
            netMoment: e.netMoment,
            circulation: e.circulation,
            staggered: e.staggered,
            tags: Array.from(e.tags ?? []),
            basin: e.basin ? basinToJSON(e.basin) : null,
        })),
    };
}

/**
 * Export lattice + params (+ optional catalog) to a pretty-printed JSON string.
 * `extra.catalog` = { randomStarts, entries } as kept by the app.
 */
export function exportJSON(lattice, params, extra = {}) {
    const doc = {
        version: VERSION,
        grid: {pitch: lattice.pitch, snap: lattice.snap, extent: lattice.extent},
        params: {
            k: params.k,
            I: params.I,
            gamma: params.gamma,
            m: params.m,
        },
        magnets: lattice.magnets.map((mg) => ({
            id: mg.id,
            cell: [mg.cell[0], mg.cell[1]],
            theta: mg.theta,
        })),
    };
    const cat = extra.catalog;
    if (cat && Array.isArray(cat.entries) && cat.entries.length) {
        doc.catalog = catalogToJSON(cat.entries, cat.randomStarts ?? 0);
    }
    return JSON.stringify(doc, null, 2);
}

function validateBasin(b, path, n) {
    if (typeof b !== 'object' || b === null) throw new Error(`${path} must be an object`);
     for (const key of ['singles', 'modes', 'randoms']) {
        if (b[key] === undefined) b[key] = [];
        if (!Array.isArray(b[key])) throw new Error(`${path}.${key} must be an array`);
        b[key].forEach((r, i) => {
            const p = `${path}.${key}[${i}]`;
            if (typeof r !== 'object' || r === null) throw new Error(`${p} must be an object`);
            if (!isNum(r.radius) || r.radius < 0) throw new Error(`${p}.radius must be a non-negative number`);
            if (r.sign !== 1 && r.sign !== -1) throw new Error(`${p}.sign must be 1 or -1`);
            if (typeof r.escaped !== 'boolean') throw new Error(`${p}.escaped must be boolean`);
            if (r.target === undefined) r.target = null;
            if (r.target !== null && !Number.isInteger(r.target))
                throw new Error(`${p}.target must be an integer or null`);
             if (r.barrier === undefined) r.barrier = null;
             if (r.barrier !== null && !isNum(r.barrier)) throw new Error(`${p}.barrier must be a number or null`);
             if (r.monotone === undefined) r.monotone = true;
             if (typeof r.monotone !== 'boolean') throw new Error(`${p}.monotone must be boolean`);
            if (key === 'singles') {
                if (!Number.isInteger(r.i) || r.i < 0 || r.i >= n)
                    throw new Error(`${p}.i must be a magnet index in [0, ${n})`);
             } else if (key === 'modes') {
                if (!Number.isInteger(r.k) || r.k < 0 || r.k >= n)
                    throw new Error(`${p}.k must be a mode index in [0, ${n})`);
                if (r.lambda === undefined) r.lambda = null;
                if (r.lambda !== null && !isNum(r.lambda)) throw new Error(`${p}.lambda must be a number or null`);
             } else {
                 if (!Number.isInteger(r.r) || r.r < 0) throw new Error(`${p}.r must be a non-negative integer`);
                 if (r.dir === undefined) r.dir = null;
                 if (r.dir !== null && (!Array.isArray(r.dir) || r.dir.length !== n || !r.dir.every(isNum)))
                     throw new Error(`${p}.dir must be null or an array of ${n} numbers`);
            }
        });
    }
     for (const key of ['minSingleRadius', 'minModeRadius', 'meanRandomRadius', 'volumeFraction']) {
        if (b[key] === undefined) b[key] = null;
        if (b[key] !== null && !isNum(b[key])) throw new Error(`${path}.${key} must be a number or null`);
    }
     b.nonMonotone = nonNegInt(b.nonMonotone, 0);
    if (b.targets === undefined) b.targets = {};
    if (typeof b.targets !== 'object' || b.targets === null || Array.isArray(b.targets))
        throw new Error(`${path}.targets must be an object {id: count}`);
    for (const [k, v] of Object.entries(b.targets)) {
        if (!/^-?\d+$/.test(k) || !Number.isInteger(v) || v < 0)
            throw new Error(`${path}.targets must map integer ids to counts`);
    }
    return b;
}

function validateCatalog(c, n) {
    if (typeof c !== 'object' || c === null) throw new Error('catalog must be an object');
    if (c.randomStarts === undefined) c.randomStarts = 0;
    if (!Number.isInteger(c.randomStarts) || c.randomStarts < 0)
        throw new Error('catalog.randomStarts must be a non-negative integer');
    if (!Array.isArray(c.entries)) throw new Error('catalog.entries must be an array');
    const ids = new Set();
    c.entries.forEach((e, idx) => {
        const p = `catalog.entries[${idx}]`;
        if (typeof e !== 'object' || e === null) throw new Error(`${p} must be an object`);
        if (!Array.isArray(e.theta) || e.theta.length !== n || !e.theta.every(isNum))
            throw new Error(`${p}.theta must be an array of ${n} angles (one per magnet)`);
        if (!isNum(e.energy)) throw new Error(`${p}.energy must be a number`);
        if (!isNum(e.lambdaMin)) throw new Error(`${p}.lambdaMin must be a number`);
        e.id = Number.isInteger(e.id) && e.id > 0 ? e.id : idx + 1;
        if (ids.has(e.id)) throw new Error(`Duplicate catalog id: ${e.id}`);
        ids.add(e.id);
        e.hits = nonNegInt(e.hits, 1);
        e.randomHits = nonNegInt(e.randomHits, 0);
        e.orbit = nonNegInt(e.orbit, 1);
        e.stabilizer = nonNegInt(e.stabilizer, 1);
        e.saddle = !!e.saddle;
        e.soft = !!e.soft;
        e.tags = Array.isArray(e.tags) ? e.tags.filter((t) => typeof t === 'string') : [];
        e.basin = e.basin === undefined || e.basin === null ? null : validateBasin(e.basin, `${p}.basin`, n);
    });
    return c;
}

/**
 * Validate a parsed document. Throws Error with a helpful message on failure.
 * Returns the validated doc (with `grid.extent` normalised to a number or null
 * and `catalog` normalised to an object or null).
 */
export function validate(doc) {
    if (typeof doc !== 'object' || doc === null) throw new Error('Root must be an object');
    if (doc.version !== VERSION)
        throw new Error(`Unsupported version: ${doc.version} (expected ${VERSION})`);
    if (typeof doc.grid !== 'object' || doc.grid === null) throw new Error('Missing "grid"');
    if (!isNum(doc.grid.pitch) || doc.grid.pitch <= 0)
        throw new Error('grid.pitch must be a positive number');
    if (typeof doc.grid.snap !== 'boolean') throw new Error('grid.snap must be boolean');
    if (doc.grid.extent === undefined || doc.grid.extent === null) {
        doc.grid.extent = null;
    } else if (!Number.isInteger(doc.grid.extent) || doc.grid.extent < 0) {
        throw new Error('grid.extent must be null or a non-negative integer');
    }
    if (typeof doc.params !== 'object' || doc.params === null) throw new Error('Missing "params"');
    for (const key of ['k', 'I', 'gamma', 'm']) {
        if (!isNum(doc.params[key])) throw new Error(`params.${key} must be a number`);
    }
    if (!Array.isArray(doc.magnets)) throw new Error('magnets must be an array');
    const seen = new Set();
    doc.magnets.forEach((mg, idx) => {
        if (typeof mg !== 'object' || mg === null) throw new Error(`magnet[${idx}] must be an object`);
        if (!Number.isInteger(mg.id)) throw new Error(`magnet[${idx}].id must be an integer`);
        if (seen.has(mg.id)) throw new Error(`Duplicate magnet id: ${mg.id}`);
        seen.add(mg.id);
        if (!Array.isArray(mg.cell) || mg.cell.length !== 2 || !isNum(mg.cell[0]) || !isNum(mg.cell[1])) {
            throw new Error(`magnet[${idx}].cell must be [number,number]`);
        }
        if (doc.grid.snap && (!Number.isInteger(mg.cell[0]) || !Number.isInteger(mg.cell[1]))) {
            throw new Error(`magnet[${idx}].cell must be [int,int] when grid.snap is true`);
        }
        if (!isNum(mg.theta)) throw new Error(`magnet[${idx}].theta must be a number`);
    });
    doc.catalog =
        doc.catalog === undefined || doc.catalog === null ? null : validateCatalog(doc.catalog, doc.magnets.length);
    return doc;
}

/**
 * Import a JSON string. Returns the validated { version, grid, params, magnets, catalog } document.
 * Throws on invalid input.
 */
export function importJSON(text) {
    let doc;
    try {
        doc = JSON.parse(text);
    } catch (e) {
        throw new Error('Invalid JSON: ' + e.message);
    }
    return validate(doc);
}