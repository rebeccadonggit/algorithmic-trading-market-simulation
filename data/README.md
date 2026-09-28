# Data

The experiments use 1-minute SPY observations for **3 March 2026** as the intraday reference price path.

The original coursework CSV is not included in this public-ready package because its redistribution terms have not yet been verified.

## Expected file

Place a licensed equivalent dataset at:

```text
data/spy_1m_2026-03-03.csv
```

Expected columns include:

```text
datetime
close
```

The original coursework file also contained:

```text
volume, vwap, open, high, low, trades
```

but the simulation runner only needs `datetime` and `close` to construct the BSE offset path.
