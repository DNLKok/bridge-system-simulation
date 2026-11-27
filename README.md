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
files remain supported and are treated as always in book.
If `run_auction` encounters an out-of-book decision, it raises a clear error
instead of silently inventing a call; provide that auction context to the
model-assistant command to get a rated recommendation.

## Generating deals with `dealer`

`DealerProvider` runs the existing Hans van Staveren `dealer` executable and
parses its PBN output. The dealer script must use an action that emits complete
PBN records, such as `action printpbn`. Its output is streamed directly from
the process; no temporary output file is needed. The full cards, board number,
dealer, vulnerability, and seed are retained, and the card hands provide the
HCP and shape profiles used by the auction.

```python
from bridge_sim import DealerProvider, load_system, run_auction

system = load_system("systems/one-no-trump-15-17.toml")
provider = DealerProvider()  # Resolves "dealer" from PATH.

for board in provider.generate("dealer/input.txt", seed=42069):
    result = run_auction(board, system, system, dealer=board.dealer)
    print(board.board_number, result.transcript, result.contract)
```

The seed is passed using dealer's `-s` option. The provider also accepts an
explicit executable path, for example `DealerProvider("/usr/games/dealer")`.
The end-to-end integration test uses
[dealer/unrestricted-single-deal.txt](dealer/unrestricted-single-deal.txt);
it skips when `dealer` is unavailable and can be pointed to an executable with
the `DEALER_EXECUTABLE` environment variable.
With seed `42069`, the passout-system integration test checks for 7NTXX after
exactly 319 auction calls.
The [dealer/south-16hcp-4432.txt](dealer/south-16hcp-4432.txt) script and
integration test also verify that the 1NT system reaches 1NT by South on a
fixed-seed 16-HCP 4-4-3-2 hand.

## Assisting during an auction

The TOML system files remain the canonical, typed definition of each bidding
rule. The assistant translates those fields into Dealer expressions when
generating a script, rather than storing a second manually maintained copy of
the rule in Dealer syntax. For example, the 1NT system's HCP and balanced-shape
fields become the corresponding `hcp(...)` and `shape(...)` condition. A pass
constrains the hand to the complement of that rule only when 1NT was a legal
call. Systems whose calls are determined solely by auction state add no hand
constraint.

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
  --partner-system systems/passout-cheapest-call.toml \
  --opponent-system systems/one-no-trump-15-17.toml \
  --n 64 \
  --seed 42069
```

`--n` defaults to 64. The assistant generates exactly that many auction-
consistent deals (or reports an error if Dealer reaches its `--generate`
limit first), runs the complete batch through DDS once, and prints JSON with
each deal's full trick table for all seats and denominations, current system
status, call ratings, and recommendation. A non-negative `--seed` is required
and reused for the whole sample. The standard
`--generate` limit is 10 billion attempts; override it for a particular run if
needed. Bid ratings use the current seat as declarer and assume a double when
the DDS result is below the contract target; otherwise the bid is undoubled.
Double and redouble rate the standing contract at the corresponding multiplier.
Pass rates the standing contract (assuming it is doubled if down) or zero when
there is no contract. Pass ratings include a TODO note because this is a
provisional model. Ratings are average IMPs versus par from the current seat's
partnership perspective; ties prefer pass, then double, redouble, and ascending
contract bids. `--dealer-executable` and `--dds-executable` can override the
default binaries.

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
boards = list(DealerProvider().generate("dealer/input.txt", seed=42069))
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
