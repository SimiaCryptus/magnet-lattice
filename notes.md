Done:

- inter-basin connectivity graph (nodes = catalog states by energy, edges = escape targets,
  width ∝ count, label/colour = energy climbed along the ray) — `src/ui/graph.js`
- random-direction basin probing → direction-averaged radius + rough ball-volume fraction
  (`ballVolumeFraction`), independent of hit counts
- warm-started boundary exploration: landings on uncatalogued minima/saddles seed a follow-up
  sweep; `relinkBasinTargets` resolves the "new" targets afterwards
- monotonicity spot-checks per ray (`monotone` flag, `nonMonotone` count, radius = smallest
  escape observed)
- state list and transition analysis split into separate cards; restyled UI

Ideas:

- barrier from a proper saddle search (nudged elastic band / dimer) instead of the ray maximum
- graph layout that avoids overlapping labels for large catalogs (force-directed x, energy y)
- export the transition graph as DOT / CSV
- run the sweep and the basin bisections in a Worker for large lattices
