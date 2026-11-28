# bridge-system-simulation

A bridge bidding system simulation environment for bidding agreement optimization

## Current implementation

The auction-only implementation includes a TOML 1NT system that bids 1NT
with 15-17 HCP and a balanced shape when legal, otherwise passing. It also
includes a passout system that passes except when passing would end the
auction, where it chooses the cheapest legal non-pass call (double, redouble,
or contract bid). See
[systems/one-no-trump-15-17.toml](systems/one-no-trump-15-17.toml) and
[systems/passout-cheapest-call.toml](systems/passout-cheapest-call.toml).

Run an auction by loading one system for each partnership and calling
`bridge_sim.run_auction(deal, north_south_system, east_west_system)`. A
profile-only `Deal` can be used for auction tests; `PBNBoard` retains full cards
and board metadata from dealer output while exposing the same hand-profile
interface. The project reuses `dealer` rather than implementing a deal
generator.

Run the tests with:

```sh
python -m unittest discover -s tests -v
```

TOML parsing uses Python's standard-library `tomllib` (Python 3.11 or newer).
Schema version 2 system files explicitly set `out_of_book` to either `"never"`
or `"when_no_rule_matches"`. An in-book 1NT system uses
`out_of_book = "never"` with `default_call = "PASS"`; a system that delegates
unmatched situations to the model assistant uses
`out_of_book = "when_no_rule_matches"` and omits `default_call`. Version 1
legacy single-rule 1NT files remain supported and are treated as always in
book. The former `strategy = "passout"` shortcut is not supported; the passout
example now stores its ordered call preferences in a module file.
Modular system manifests use `kind = "system"` and list ordered, version-pinned
modules by ID and relative TOML path. The first module rule that matches and
selects a legal call wins; `on_no_match` explicitly supplies the fallback.
The 1NT and passout examples remain separate systems, each referencing its own
module. Their rules are data in [systems/modules/](systems/modules/); the
loader interprets the shared rule vocabulary. Existing single-system TOML
1NT files remain loadable.
The [traditional preempts module](systems/modules/traditional-preempts.toml)
demonstrates suit-aware rules and can be loaded as the
[traditional preempts system](systems/traditional-preempts.toml). It covers
suit bids from 2D through 5D (excluding notrump and 2C), maps 6/7/8/9+
cards to the 2/3/4/5 level, and prefers the longest qualifying suit, breaking
ties using the S-H-D-C order declared in the module TOML. Hand profiles from
cards include per-suit HCP; a
profile-only hand without per-suit HCP cannot evaluate a rule that requires it.
The [simple overcalls module](systems/modules/simple-overcalls.toml) handles
1C-2S suit openings, including calls that may be artificial in a particular
opponent system. Its standalone system is in
[systems/simple-overcalls.toml](systems/simple-overcalls.toml); in a composed
system, place specialized opening-response modules before it so they can take
priority. The module chooses the longest qualifying non-jump suit overcall,
then prefers S-H-D-C. Suit overcalls show 5+ cards; they require 9+ HCP at the
1-level, 12+ with five cards or 10+ with six at the 2-level, and 13+ with five
cards or 11+ with six at the 3-level, with a 17-HCP maximum. It also defines
balanced notrump overcalls/doubles and takeout doubles. Card-derived profiles
include the suit-honor details needed to evaluate stopper rules.
The [standard openings module](systems/modules/standard-openings.toml) opens
when HCP plus the lengths of the two longest suits total at least 20, then
handles balanced notrump and strong openings before choosing a longest-suit
1-level opening. Its standalone system is
[systems/standard-openings.toml](systems/standard-openings.toml).

## Standard bidding-system test

[systems/standard-test.toml](systems/standard-test.toml) composes standard
openings, traditional preempts, and simple overcalls (including its takeout
doubles), in that priority order. The same composite system is assigned to
North-South and East-West in [simulation/standard-test.toml](simulation/standard-test.toml).
Edit that settings file to change the deal count, seed, dealer, vulnerability,
systems, or PDF output. Advanced out-of-book logic is disabled and the runner
rejects settings that enable it. Run the test with:

```sh
python -m bridge_sim.standard_test
```

The default run generates 16 deals with seed `20261128` and writes
[reports/standard-test.pdf](reports/standard-test.pdf). The PDF starts with
the systems and settings, then shows each complete deal as a compass hand
diagram beside its complete auction. PDF rendering requires Chromium on `PATH`.

If `run_auction` encounters an out-of-book decision, it raises a clear error
instead of silently inventing a call; provide that auction context to the
model-assistant command to get a rated recommendation.

## Generating deals with `dealer`

`DealerProvider` runs the workspace-local executable at `bin/dealer` by
default and parses its PBN output. This bundled binary is for Linux x86-64.
The dealer script must use an action that emits complete PBN records, such as
`action printpbn`. Its output is streamed directly from the process; no
temporary output file is needed. The full cards, board number, dealer,
vulnerability, and seed are retained, and the card hands provide the HCP and
shape profiles used by the auction.

```python
from bridge_sim import DealerProvider, load_system, run_auction

system = load_system("systems/one-no-trump-15-17.toml")
provider = DealerProvider("bin/dealer")

for board in provider.generate("dealer/input.txt", seed=42069):
    result = run_auction(board, system, system, dealer=board.dealer)
    print(board.board_number, result.transcript, result.contract)
```

The seed is passed using the local executable's `-s` option. `DealerProvider`
also accepts an explicit executable path when using a compatible build. The
end-to-end integration test uses
[dealer/unrestricted-single-deal.txt](dealer/unrestricted-single-deal.txt) and
the workspace-local binary.
With seed `42069`, the passout-system integration test checks for 7NTXX after
exactly 319 auction calls.
The [dealer/south-16hcp-4432.txt](dealer/south-16hcp-4432.txt) script and
integration test also verify that the 1NT system reaches 1NT by South on a
fixed-seed 16-HCP 4-4-3-2 hand.

## Assisting during an auction

The TOML module files are the canonical, typed definition of each bidding
rule, while system manifests compose ordered module versions and define a
fallback. Rules specify conditions and either a call or an ordered `try_calls`
list; the Python code interprets these generic fields rather than encoding the
sample bidding agreements. The assistant translates the same TOML conditions
into Dealer expressions when generating a script. For example, the 1NT
module's HCP range and shape list become `hcp(...)` and `shape(...)`
conditions. A pass constrains the hand to the complement of that rule only
when 1NT was legal. Auction-only conditions add no hand constraint.
Suit-aware conditions use the same Dealer forms, for example
`hcp(north, spades)` and `spades(north)`.

Generate a script from the perspective of the seat currently due to call:

```sh
python -m bridge_sim.auction_assistant \
  --seat W \
  --dealer N \
  --vulnerability NS \
  --hand 'AKQ.JT9.876.5432' \
  --auction '1NT PASS PASS' \
  --partner-system systems/passout-cheapest-call.toml \
  --opponent-system systems/one-no-trump-15-17.toml \
  --produce 1 \
  --output /tmp/auction-assistance.txt
dealer /tmp/auction-assistance.txt
```

The hand is written in PBN suit order (spades.hearts.diamonds.clubs). The
generated Dealer script uses `predeal` for that exact hand, validates that the
auction is legal and consistent with the supplied systems, and includes the
translated hand constraints for partner and opponents. `--auction` lists calls
from the dealer onward; use `P` or `PASS`, `X`, `XX`, and bids such as `1NT`.
The generated script is also printed to standard output when `--output` is
omitted.

Run the model assistant over a Monte Carlo sample and solve every generated
board with the local DDS executable. It reports an average IMP rating for every
legal call, but uses the system's explicit call when in book and the highest-
rated call only when the system is out of book:

```sh
python -m bridge_sim.model_assistant \
  --seat W \
  --dealer N \
  --vulnerability NS \
  --hand 'AKQ.JT9.876.5432' \
  --auction '1NT PASS PASS' \
  --partner-system systems/one-no-trump-15-17.toml \
  --opponent-system systems/one-no-trump-15-17.toml \
  --n 8 \
  --advanced \
  --seed 42069
```

`--n` defaults to 64. The assistant generates exactly that many auction-
consistent deals (or reports an error if Dealer reaches its `--generate`
limit first), runs the complete batch through DDS once, and prints JSON with
each deal's full trick table for all seats and denominations, current system
status, call ratings, and recommendation. A non-negative `--seed` is required
and determines the sample reproducibly; advanced mode derives separate seeds
for its nested samples. Advanced mode applies only when the current hand is
out of book. It keeps the existing pass rating, but for these calls:

- legal double or redouble;
- the cheapest legal bid in each strain; and
- the game bids 3NT, 4H, 4S, 5C, and 5D.

Overlapping calls are evaluated once. All other calls retain their existing
direct rating. For each selected call, it appends that call and a pass by the
left-hand opponent, then runs the basic assistant once from partner's
perspective using the partner's hand from one separate, auction-matched outer
deal and `N` compatible random deals. The mean rating for partner's recommended
call becomes the value for that candidate, with no sign change. A separate
basic `N`-deal sample continues to provide pass and other direct ratings.
This is one recursion level only. Since advanced mode evaluates each selected
call for every outer deal, start with a smaller `--n` if runtime is a concern.
Without `--advanced`, the assistant retains the existing rating behavior. The
standard `--generate` limit is 10 billion attempts; override it for a particular
run if needed. Bid ratings use the current seat as declarer and assume a double
when the DDS result is below the contract target; otherwise the bid is undoubled.
Double and redouble rate the standing contract at the corresponding multiplier.
Pass rates the standing contract (assuming it is doubled if down) or zero when
there is no contract. Pass ratings include a TODO note because this is a
provisional model. Ratings are average IMPs versus par from the current seat's
partnership perspective; ties prefer pass, then double, redouble, and ascending
contract bids. `--dealer-executable` and `--dds-executable` can override the
default binaries.

The real-binary `N=16` advanced performance test is opt-in because it runs many
nested samples. Run it with
`RUN_ADVANCED_PERFORMANCE_TESTS=1 python -m unittest tests.test_model_assistant_performance`;
it uses an empty auction and fixed hand and prints the elapsed time.

## Evaluating contracts with DDS

`run_auction_batch` completes every auction before invoking the local
`TablesPBN4` executable once with all contracted deals. The signed
`AuctionResult.trick_delta` is declarer tricks minus the contract target:
negative values are undertricks, positive values are overtricks, and zero
means exactly made. Passed-out boards have `trick_delta=None`.
The solver selects the trick-table entry for the contract's declarer and
denomination, so the correct opening-lead perspective is handled by DDS.

```python
from bridge_sim import DealerProvider, load_system, run_auction_batch

system = load_system("systems/one-no-trump-15-17.toml")
boards = list(
    DealerProvider("bin/dealer").generate("dealer/input.txt", seed=42069)
)
results = run_auction_batch(boards, system, system)

for board, result in zip(boards, results, strict=True):
    print(board.board_number, result.contract, result.trick_delta)
```

By default, the solver uses
`Dealer-Version-2--main/DDS_Demo/SolveTables/TablesPBN4` in this workspace.
An alternate executable path can be supplied to `TablesPBN4Solver`.
From the complete trick table, the simulator calculates a `ParResult` using the
board's dealer and vulnerability. `result.par_result` contains the par contract
(including declarer and double/redouble status), signed trick delta, and
North-South score. A passed-out par has no contract or trick delta and scores
zero. The convenience properties `result.par_contract` and
`result.par_score_ns` expose the contract and score directly. `score_ns` is the
attained score, and `score_vs_par` is `score_ns - par_score_ns` (positive means
North-South outscored par). `imps_vs_par` converts that NS-oriented difference
to IMPs using the standard IMP scale; positive values favor North-South.

## Calculating duplicate scores

`calculate_score_ns(contract, vulnerability, trick_delta)` returns the standard
duplicate score from North-South's perspective. It accounts for the declarer's
vulnerability, denomination, contract level, doubles/redoubles, overtricks,
undertrick penalties, game, slam, and insult bonuses.

```python
from bridge_sim import calculate_score_ns

if result.contract is not None and result.trick_delta is not None:
    score_ns = calculate_score_ns(
        result.contract,
        board.vulnerability,
        result.trick_delta,
    )
```
