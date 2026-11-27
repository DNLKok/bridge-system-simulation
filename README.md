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
From the complete trick table, the simulator also calculates `par_score_ns`
using the board's dealer and vulnerability. `score_ns` is the attained score,
and `score_vs_par` is `score_ns - par_score_ns` (positive means North-South
outscored par).

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
