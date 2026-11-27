# bridge-system-simulation
A bridge bidding system simulation environment for bidding agreement optimization

## Current implementation

The first auction-only implementation supports a TOML system with one rule:
bid 1NT when the hand has 15-17 HCP, has a balanced shape, and 1NT is legal;
otherwise pass. See
[systems/one-no-trump-15-17.toml](systems/one-no-trump-15-17.toml).

Run an auction by loading one system for each partnership and calling
`bridge_sim.run_auction(deal, north_south_system, east_west_system)`. The
`Deal` input contains one `HandProfile` per seat, with HCP and four suit
lengths. It is an auction-only projection of an existing deal; this project
does not generate deals or cards, or parse card-based deal formats yet.

Run the tests with:

```sh
python -m unittest discover -s tests -v
```

TOML parsing uses Python's standard-library `tomllib` (Python 3.11 or newer).
