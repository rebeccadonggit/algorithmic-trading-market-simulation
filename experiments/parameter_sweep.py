

import os
import pandas as pd

from src.market_runner import run_market_session_with_offset, RESULTS_DIR, DEFAULT_DATA_FILE
from experiments.backtest import summarise_run


OFFSET_FILE = DEFAULT_DATA_FILE
COST_PER_TRADE = 1.0
TUNING_RESULTS_DIR = RESULTS_DIR


def run_one_tuning_trial(bb_window, bb_k, seed, market_noise='wide', cost_per_trade=COST_PER_TRADE):
    """
    Run one MMM02 tuning trial with a specific Bollinger parameter pair.
    Returns one summary row.
    """
    trial_id = f'tuning_mmm02_w{bb_window}_k{str(bb_k).replace(".", "p")}_{market_noise}_seed_{seed:03d}'

    trial_prefix = run_market_session_with_offset(
        strategy_name='MMM02',
        trial_id=trial_id,
        random_seed=seed,
        price_offset_filename=OFFSET_FILE,
        market_noise=market_noise,
        mm_params={'bb_window': bb_window, 'bb_k': bb_k},
    )

    row = summarise_run(trial_prefix, 'MMM02', seed)
    row['bb_window'] = bb_window
    row['bb_k'] = bb_k
    row['market_noise'] = market_noise
    row['cost_per_trade'] = cost_per_trade
    row['n_trades'] = row['buys'] + row['sells']
    row['total_transaction_cost'] = row['n_trades'] * cost_per_trade
    row['net_total_equity_pnl'] = row['total_equity_pnl'] - row['total_transaction_cost']
    return row



def run_parameter_sweep(
    bb_windows=(10, 20, 30),
    bb_ks=(1.5, 2.0, 2.5),
    seeds=tuple(range(1, 6)),
    market_noise='wide',
    cost_per_trade=COST_PER_TRADE,
):
    """
    Run a small MMM02 Bollinger parameter sweep.
    Saves both raw trial results and an aggregated summary.
    """
    results = []

    for bb_window in bb_windows:
        for bb_k in bb_ks:
            for seed in seeds:
                print(
                    f'Running MMM02 tuning: bb_window={bb_window}, bb_k={bb_k}, '
                    f'seed={seed}, market_noise={market_noise}, cost_per_trade={cost_per_trade} ...'
                )
                row = run_one_tuning_trial(
                    bb_window,
                    bb_k,
                    seed,
                    market_noise=market_noise,
                    cost_per_trade=cost_per_trade,
                )
                results.append(row)

    df = pd.DataFrame(results)

    raw_file = os.path.join(TUNING_RESULTS_DIR, 'tuning_mmm02_raw_results.csv')
    df.to_csv(raw_file, index=False)

    summary = (
        df.groupby(['market_noise', 'bb_window', 'bb_k', 'cost_per_trade'])
        .agg(
            mean_pnl=('total_equity_pnl', 'mean'),
            std_pnl=('total_equity_pnl', 'std'),
            mean_net_pnl=('net_total_equity_pnl', 'mean'),
            std_net_pnl=('net_total_equity_pnl', 'std'),
            mean_total_transaction_cost=('total_transaction_cost', 'mean'),
            mean_roundtrips=('roundtrips', 'mean'),
            std_roundtrips=('roundtrips', 'std'),
            mean_expectancy=('expectancy', 'mean'),
            std_expectancy=('expectancy', 'std'),
            mean_win_rate=('win_rate', 'mean'),
            mean_end_inventory=('end_inventory', 'mean'),
            end_inventory_positive_ratio=('end_inventory', lambda s: (s > 0).mean()),
            n_trials=('seed', 'count'),
        )
        .reset_index()
        .sort_values(['market_noise', 'mean_net_pnl', 'mean_expectancy'], ascending=[True, False, False])
    )

    summary_file = os.path.join(TUNING_RESULTS_DIR, 'tuning_mmm02_summary.csv')
    summary.to_csv(summary_file, index=False)

    print('\nSaved raw tuning results to:')
    print(raw_file)
    print('\nSaved tuning summary to:')
    print(summary_file)
    print('\nTuning summary:')
    print(summary)

    return df, summary


if __name__ == '__main__':
    run_parameter_sweep(
        bb_windows=(10, 20, 30),
        bb_ks=(1.5, 2.0, 2.5),
        seeds=tuple(range(1, 6)),
        market_noise='wide',
        cost_per_trade=COST_PER_TRADE,
    )