# Bridge System Simulation: Design

## Goal

Build a Python environment that can run many bridge deals through configurable
bidding systems, evaluate the resulting contracts with a double-dummy solver,
and compare systems on the same reproducible set of boards.

The design separates the bidding rules from the auction, deal source, solver,
scoring, and experiment runner. This makes systems easy to edit and swap, and
lets the native C++ solver and existing `dealer` generator remain in use while
the Python parts are developed.

## Main components

```text
bridge_sim/
  model.py       Cards, hands, boards, calls, contracts
  auction.py     Legal calls, turn order, auction completion
  systems.py     System-file loading, validation, rule selection
  deals.py       Deal-provider interface and `dealer` adapter
  dds.py         Double-dummy solver adapter
  scoring.py     Duplicate scoring and score perspective
  runner.py      Experiments, repetitions, result writing
systems/
  modules/       Reusable bidding-rule modules
  *.toml         Complete system manifests
tests/
  fixtures/      Small, fixed deals and bidding situations
```

These are logical boundaries, not a requirement to create every file at once.
Start with a small end-to-end implementation and split modules as the code
grows.

### Auction and information boundaries

For each turn, a system receives the bidder's hand, the public auction, dealer,
vulnerability, and legal calls. It must not receive the other three hands.
Keep this interface explicit: it prevents accidental use of hidden information
and makes a decision reproducible from its inputs.

The auction engine owns bridge call legality, turn order, and termination.
Bidding systems choose among legal calls; they do not implement auction
mechanics themselves. The initial implementation supports pass, contract
bids, doubles, and redoubles. Invalid calls and ambiguous rules should produce
clear validation errors, not be silently ignored.

### Human-readable systems

Use versioned TOML as the initial canonical format. TOML is editable and
reviewable in source control, and Python's standard library can parse it
without an extra dependency. The initial implementation uses `tomllib` and
therefore requires Python 3.11 or newer. Avoid embedding arbitrary Python
expressions in a system file. Instead, implement a growing set of typed
conditions (for example HCP range, hand shape, suit length, auction pattern,
vulnerability, and whether either partnership has bid).

A simplified rule might look like this:

```toml
schema_version = 1
id = "basic"
version = "1.0.0"
name = "Basic example"
default_call = "PASS"

[[rules]]
id = "open-1nt"
priority = 100
call = "1NT"
explanation = "15-17 HCP, balanced"

[rules.when.auction]
contract_exists = false
own_side_has_bid = false

[rules.when.hand]
min_hcp = 15
max_hcp = 17
balanced = true
```

This is an illustrative schema, not a claim that the first implementation
should support every condition shown. Make the supported condition vocabulary
explicit and validate it. Give rules stable IDs so that an override or a
semantic diff can identify a rule even if its explanatory text changes.

If several rules match, use explicit priority and report unresolved ties.
Make fallback behavior explicit; count and report how often it is used so
missing system coverage is visible. Store explanations alongside actions so
auction records can show why a call was selected.

## Reusing and composing many systems

Treat a complete bidding system as a named, versioned policy, and shared
agreements as separately versioned rule modules. The implemented format uses a
system manifest with an ordered list of module references; each reference pins
an ID and version and points to a TOML module file relative to the manifest.
Modules are tried in order, and the first one to return a legal decision wins.
The manifest must declare its `on_no_match` fallback. A system can reuse any
number of versioned modules without embedding their rules in Python.

The current 1NT and passout examples are separate systems, each referencing
its own module. They are intentionally not combined.

The loader checks module IDs and pinned versions, rejects duplicate module
references and invalid schemas, and preserves support for the older
single-system TOML format by translating legacy definitions into the same
generic rule representation. Module TOMLs contain ordered rules, conditions,
and actions. The interpreter supports HCP bounds, shape sets, passout-seat
conditions, a fixed call, or an ordered `try_calls` preference list. New
agreements can be added by authoring module data using those primitives;
adding a new condition primitive requires extending the generic interpreter.
The runtime decision context includes hand, auction, legal calls, seat, dealer,
and vulnerability, allowing future rules to use only information available to
that bidder.

A module rule stores the agreement itself:

```toml
[[rules]]
id = "open-1nt"
call = "1NT"

[rules.when]
min_hcp = 15
max_hcp = 17
shapes = ["4333", "4432", "5332"]
```

`try_calls` lists call tokens in preference order and the interpreter selects
the first legal one. Auction-only rules can use `passout_seat`; for example, a
passout module stores both that condition and its ordered call list in TOML.
No specific 1NT or passout policy is implemented as a Python module class.

Explicit named overrides, rule priorities, and situation-based delegation to
other systems are possible future extensions; the current format does not
silently implement them.

Also support a system that delegates a decision to another system in a defined
situation. For example, a future composite policy could use one pinned system
for uncontested auctions and another for competitive auctions:

```toml
schema_version = 1
id = "hybrid"
version = "1.0.0"

[[strategy]]
use = "standard@1.4.0"
[strategy.when]
competitive = false

[[strategy]]
use = "aggressive@2.1.0"
[strategy.when]
competitive = true

[[strategy]]
use = "conservative@1.2.0"
otherwise = true
```

The exact syntax can evolve, but the semantics should be deliberate:

- Resolve a strategy for each decision using only that bidder's hand and
  public information available at that point in the auction.
- Pin each referenced system or module to an immutable version, not a moving
  name such as `latest`.
- Define route ordering, overlap validation, and an explicit `otherwise`
  behavior. Do not silently fall through to a different policy.
- Detect circular references and report the full reference chain.
- Record every resolved component and override when the composite is loaded.

For a small variation, prefer a module plus a few explicit overrides. For a
substantially different policy, create a new system that reuses only the
agreements that genuinely remain shared. This keeps both reuse and differences
visible during review.

### Comparing systems

Keep system files and modules in version control. Provide tooling to:

1. Show a text diff of the source files.
2. Show a semantic diff of the resolved rules: added, removed, and changed
   conditions, calls, priorities, and routes.
3. Explain which rule selected a call for a supplied hand and auction.
4. Run shared fixed situations as regression tests against multiple systems.

The semantic diff should compare normalized, resolved rules rather than YAML
formatting alone. This makes it easier to answer both “what changed?” and
“which situations behave differently?” Reusable modules reduce duplicated
agreements; regression situations check that a refactor did not change behavior
unexpectedly.

## Deals, double-dummy evaluation, and scoring

Keep a deal-provider interface in Python and wrap the existing `dealer`
program. Its `printpbn` action can emit multiple boards in one process, so
stream PBN records from stdout rather than starting an external process per
board or writing a temporary output file. Parse and retain the cards, board
number, dealer, vulnerability, and seed; derive HCP and shape from each parsed
hand for bidding. Reimplement deal generation only if measurements or needed
features justify it. Record the generator version, input script, parameters,
and seed.

Keep the C++ double-dummy solver behind a small Python adapter. Prefer a native
library binding (for example `pybind11`, or a stable C interface called from
Python) over rewriting a working solver. If only a command-line executable is
available, batch requests to limit process-start and input/output overhead.

After the auction, use the solver's trick result for the final declarer and
strain, then apply duplicate scoring with the board's vulnerability, contract
level, and double/redouble status. A passed-out board scores zero. Store scores
from a consistent perspective, such as North-South, and explicitly reverse the
sign when reporting from the other partnership's perspective.

Double-dummy play assumes every hand is visible. It is a consistent measure of
contract outcome for this experiment, but it is not an estimate of ordinary
single-dummy human play.

Cache solver results by deal. The same deal can be used to compare many
systems, while its double-dummy trick table does not change with the bidding
system. Benchmark before optimizing further; then consider native batching,
parallel execution, or a more efficient deal provider.

## Experiments and audit trail

Make every simulation an immutable, named run. Save a run manifest containing:

- Run ID and UTC start time.
- Code revision and simulator version.
- System IDs and versions, plus hashes of the fully resolved system and all
  referenced modules.
- Deal-generator identity, version, parameters, and random seed.
- Double-dummy solver identity/build and relevant options.
- Board count, dealer/vulnerability setup, system seat assignments, and
  scoring method.

The resolved-content hashes matter: a label alone cannot identify what was
actually used if files later change. Preserve the resolved system snapshot (or
enough version-controlled references to reconstruct it) with the run.

Write both aggregate results and board-level data. A practical first version
can use:

```text
runs/<run-id>/
  manifest.json
  summary.csv
  boards.csv
  auctions.jsonl
```

Each board record should include its deal ID, dealer, vulnerability, systems
seated North-South and East-West, auction transcript, final contract, declarer,
double-dummy tricks, and score. Auction records should identify the selected
rule and resolved system component for each call. CSV is easy to inspect;
Parquet or SQLite can be added if result volume makes CSV unwieldy.

Compare systems on the same boards, and rotate which system occupies each
partnership where appropriate. Report raw score per board and IMPs per board,
along with win/loss counts; for smaller samples, include an uncertainty
estimate. Record the seat rotation and score perspective in the manifest so
that a result can be interpreted later.

## Build sequence

1. Define and test cards, hands, boards, calls, contracts, and duplicate
   scoring.
2. Implement auction legality and termination for passes, bids, doubles, and
   redoubles, with fixed-deal tests.
3. Load one simple YAML system and complete an auction while enforcing the
   information boundary.
4. Wrap the existing deal generator and C++ solver behind Python interfaces.
5. Run a small seeded matchup, save the manifest and board-level results, and
   verify that repeating it reproduces the same outcomes.
6. Add system modules, explicit overrides, composition, semantic diffs, and
   regression situations.
7. Profile representative large runs before optimizing or adding parallelism.

This order delivers a working experiment early while making system reuse,
comparison, and run provenance core design concerns rather than later
add-ons.
