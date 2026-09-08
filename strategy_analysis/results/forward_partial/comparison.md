# Partial forward test - ORB Scenario 3 (data-available setups)

- In-sample TRUE-timeframe (['2025-08-06', '2026-01-29']): 74 trades, WR 48.65%, net 51.0R
- FORWARD (['2026-02-05', '2026-05-21']): 37 trades, WR 43.24%, net 15.0R

| Setup | IS-trueTF n | IS-trueTF WR | IS-trueTF R | FWD n | FWD WR | FWD R |
|---|---|---|---|---|---|---|
| Thursday|EURUSD|05:00 PM | 24 | 45.8 | 9.0 | 16 | 43.8 | 5.0 |
| Thursday|GBPUSD|09:15 AM | 24 | 33.3 | 0.0 | 11 | 45.5 | 4.0 |
| Wednesday|GBPUSD|08:30 PM | 26 | 65.4 | 42.0 | 10 | 40.0 | 6.0 |

## Full trade log (forward)

| date       | day_of_week   | pair   | orb_time_ist   | direction   |   rr_ratio | result   |   pnl_pips | exit_reason   |
|:-----------|:--------------|:-------|:---------------|:------------|-----------:|:---------|-----------:|:--------------|
| 2026-02-05 | Thursday      | EURUSD | 05:00 PM       | LONG        |          2 | LOSS     |       -6.9 | SL Hit        |
| 2026-02-12 | Thursday      | EURUSD | 05:00 PM       | LONG        |          2 | LOSS     |       -7.9 | SL Hit        |
| 2026-02-19 | Thursday      | EURUSD | 05:00 PM       | SHORT       |          2 | WIN      |       15.8 | TP Hit        |
| 2026-02-26 | Thursday      | EURUSD | 05:00 PM       | LONG        |          2 | LOSS     |       -6.3 | SL Hit        |
| 2026-03-05 | Thursday      | EURUSD | 05:00 PM       | LONG        |          2 | LOSS     |       -9.7 | SL Hit        |
| 2026-03-12 | Thursday      | EURUSD | 05:00 PM       | SHORT       |          2 | WIN      |       30   | TP Hit        |
| 2026-03-12 | Thursday      | GBPUSD | 09:15 AM       | SHORT       |          2 | LOSS     |      -11   | SL Hit        |
| 2026-03-18 | Wednesday     | GBPUSD | 08:30 PM       | SHORT       |          3 | LOSS     |      -16.6 | SL Hit        |
| 2026-03-19 | Thursday      | EURUSD | 05:00 PM       | LONG        |          2 | LOSS     |       -9.7 | SL Hit        |
| 2026-03-19 | Thursday      | GBPUSD | 09:15 AM       | SHORT       |          2 | WIN      |       25   | TP Hit        |
| 2026-03-25 | Wednesday     | GBPUSD | 08:30 PM       | SHORT       |          3 | WIN      |        4.5 | EOD Close     |
| 2026-03-26 | Thursday      | EURUSD | 05:00 PM       | SHORT       |          2 | LOSS     |      -18.3 | SL Hit        |
| 2026-03-26 | Thursday      | GBPUSD | 09:15 AM       | LONG        |          2 | LOSS     |       -6.6 | SL Hit        |
| 2026-04-01 | Wednesday     | GBPUSD | 08:30 PM       | SHORT       |          3 | LOSS     |      -20.3 | SL Hit        |
| 2026-04-02 | Thursday      | EURUSD | 05:00 PM       | LONG        |          2 | WIN      |       33.4 | TP Hit        |
| 2026-04-02 | Thursday      | GBPUSD | 09:15 AM       | SHORT       |          2 | WIN      |       36.6 | TP Hit        |
| 2026-04-08 | Wednesday     | GBPUSD | 08:30 PM       | SHORT       |          3 | WIN      |       34.3 | EOD Close     |
| 2026-04-09 | Thursday      | EURUSD | 05:00 PM       | LONG        |          2 | LOSS     |      -11   | SL Hit        |
| 2026-04-09 | Thursday      | GBPUSD | 09:15 AM       | LONG        |          2 | LOSS     |      -18.1 | SL Hit        |
| 2026-04-15 | Wednesday     | GBPUSD | 08:30 PM       | LONG        |          3 | LOSS     |      -12.7 | SL Hit        |
| 2026-04-16 | Thursday      | EURUSD | 05:00 PM       | SHORT       |          2 | WIN      |        0.2 | EOD Close     |
| 2026-04-16 | Thursday      | GBPUSD | 09:15 AM       | SHORT       |          2 | WIN      |       13.2 | TP Hit        |
| 2026-04-22 | Wednesday     | GBPUSD | 08:30 PM       | LONG        |          3 | LOSS     |       -8.5 | EOD Close     |
| 2026-04-23 | Thursday      | EURUSD | 05:00 PM       | LONG        |          2 | WIN      |       19.8 | TP Hit        |
| 2026-04-23 | Thursday      | GBPUSD | 09:15 AM       | LONG        |          2 | WIN      |       12.2 | TP Hit        |
| 2026-04-29 | Wednesday     | GBPUSD | 08:30 PM       | SHORT       |          3 | LOSS     |       -0.8 | EOD Close     |
| 2026-04-30 | Thursday      | EURUSD | 05:00 PM       | SHORT       |          2 | LOSS     |      -15.9 | SL Hit        |
| 2026-04-30 | Thursday      | GBPUSD | 09:15 AM       | SHORT       |          2 | WIN      |       12   | TP Hit        |
| 2026-05-06 | Wednesday     | GBPUSD | 08:30 PM       | SHORT       |          3 | WIN      |        0.1 | EOD Close     |
| 2026-05-07 | Thursday      | EURUSD | 05:00 PM       | LONG        |          2 | LOSS     |       -8.8 | SL Hit        |
| 2026-05-07 | Thursday      | GBPUSD | 09:15 AM       | LONG        |          2 | LOSS     |       -7.3 | SL Hit        |
| 2026-05-13 | Wednesday     | GBPUSD | 08:30 PM       | LONG        |          3 | WIN      |        0.8 | EOD Close     |
| 2026-05-14 | Thursday      | EURUSD | 05:00 PM       | SHORT       |          2 | WIN      |       15.8 | TP Hit        |
| 2026-05-14 | Thursday      | GBPUSD | 09:15 AM       | LONG        |          2 | LOSS     |       -6   | SL Hit        |
| 2026-05-20 | Wednesday     | GBPUSD | 08:30 PM       | LONG        |          3 | LOSS     |      -12.4 | EOD Close     |
| 2026-05-21 | Thursday      | EURUSD | 05:00 PM       | SHORT       |          2 | WIN      |       22.2 | TP Hit        |
| 2026-05-21 | Thursday      | GBPUSD | 09:15 AM       | LONG        |          2 | LOSS     |       -7.3 | SL Hit        |