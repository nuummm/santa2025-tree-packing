# Design notes

## Problem (summary in my own words)

For every n = 1..200, place n copies of a fixed 15-vertex "Christmas tree" polygon
(any position, any rotation, no two interiors overlapping; touching is allowed) so
that the axis-aligned square containing all of them is as small as possible. The
score is the sum over n of side(n)² / n; lower is better. A submission lists
`x, y, deg` for each tree as strings prefixed with `s`, and coordinates must stay in
[-100, 100]. Source: the competition's Overview, Evaluation and Data pages
(<https://www.kaggle.com/competitions/santa-2025>).

Two facts shape the solver:

* The 200 sub-problems are independent, so they parallelise trivially and can be
  improved and selected one n at a time.
* A tree has area A = 0.245625, so side(n)² ≥ n·A and the score is at least
  200·A = 49.125. Density n·A / side² is the natural per-n quality measure.

## Geometry

The tree is three stacked tiers and a trunk. It is not convex, so it is split into
four convex pieces (top triangle, two trapezoids, trunk rectangle). Tests check that
the pieces' union equals the polygon, that each piece is convex, the area, and that
every vertex lies within 0.8 of the trunk-top origin (used as a bounding disc).

## Two-stage correctness

| Stage | Where | How | Role |
|---|---|---|---|
| fast test | `collision.py` (Numba) | disc → box → separating-axis test over the 4×4 piece pairs | used millions of times per second during search |
| final gate | `validate.py` (shapely) | values parsed from the written strings as `Decimal`, scaled by 1e18, `intersects and not touches`, every n | decides what is written |

The fast test is **conservative**: it calls a pair clear only if some edge normal (or
box axis, or the disc test) separates them by more than `eps = 1e-9`. Contact and
near-contact count as collisions. A positive gap along any axis implies the polygons
are strictly apart, so a layout the fast test accepts is also accepted by an exact
test, up to the rounding of the coordinates themselves (~1e-15 here, far below
`eps`). The tests check this implication on pairs generated right at contact, and
check full agreement with shapely away from contact. This is an empirical check plus
an argument, not a proof, which is why the final gate exists and why only layouts
that pass it are saved.

The 1e18 scaling in the gate is not about precision; it puts coordinates far from 1
so that any absolute tolerance inside the geometry library is negligible relative to
the trees. The gate follows the rules as published, but it is my own code, not the
competition's metric, so agreement with the leaderboard is checked by submitting.

## Search

| Stage | Module | Idea |
|---|---|---|
| double lattice | `lattice.py` | A tree and its half-turned copy form a cell; cells repeat on a 2-D lattice. Scan (θ, u, v), bisect the periods, minimise cell area; then cut the n trees that fit in the smallest square window. |
| annealing | `anneal.py` | Moves: translate, rotate, or both, with step sizes shrinking geometrically. 70 % of moves pick a tree on the boundary of the union box. Energy `max(w, h) + 0.1·(w + h)`. Metropolis test first (O(n)), collision test only if it passes. |
| compaction | `compact.py` | Scale the layout towards its centre, then push trees inward one by one; halve the step when a sweep stops helping. Kept only if the square side shrinks. |
| remove-one | `initializers.py`, `pipeline.py` | Walk n downward: dropping the best boundary tree from the n+1 layout (then compacting) sometimes beats the direct n layout. |
| ensemble | `ensemble.py` | Several seeds; per n keep the smallest layout that passes the final gate. |

**Budget in moves, not seconds.** The annealer runs `moves_per_tree × max(n, 30)`
moves and seeds NumPy's RNG inside the compiled function from `(seed, n)`. A run is a
pure function of its arguments: the same command gives a byte-identical
`submission.csv` whatever the number of workers (there is a test for this).

**Why trees on the boundary.** Only trees that touch the union box can shrink it;
picking them more often concentrates moves where the objective can change.

**Why the extra `w + h` term.** With several trees on the binding side, moving one
of them inward does not change `max(w, h)`, so the pure objective is flat exactly where
progress is being made. The small `w + h` term gives those moves a slope.

## Known techniques used, and references

| Technique | Where | Reference |
|---|---|---|
| Separating axis theorem for convex polygons | `collision.py` | C. Ericson, *Real-Time Collision Detection*, Morgan Kaufmann, 2004, ch. 5 |
| Decomposing a non-convex shape into convex pieces | `geometry.py` | J. A. Bennell, J. F. Oliveira, "The geometry of nesting problems: a tutorial", *EJOR* 184(2), 2008 |
| Double-lattice packing (a body and its half-turn) | `lattice.py` | G. Kuperberg, W. Kuperberg, "Double-lattice packings of convex bodies in the plane", *Discrete Comput. Geom.* 5, 1990 |
| Simulated annealing | `anneal.py` | S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi, "Optimization by simulated annealing", *Science* 220, 1983 |
| Deriving the n layout from the n+1 layout | `initializers.py` | an idea widely discussed among participants of the competition |

The parts I designed for this repository are the combination: the conservative
fast test paired with an independent shapely gate on every n, the move-count budget
with in-kernel seeding for reproducibility, the compaction pass, and the validated
per-n ensemble.

## Limitations and next steps

* The annealer only accepts overlap-free states. Allowing temporary overlap with a
  penalty usually explores dense arrangements better.
* Small n (roughly n ≤ 20) start from a lattice cut-out, which is a poor start for
  them; dedicated starts or exhaustive symmetric layouts would help.
* The budget is uniform per tree. Spending more moves where the per-n gap to the
  density curve is largest is the obvious next experiment.
