import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.market_runner import (
    RESULTS_DIR,
    DEFAULT_DATA_FILE,
    run_market_session_with_offset,
    extract_mm_trades_from_blotter,
)

COURSEWORK_OFFSET_FILE = DEFAULT_DATA_FILE
FINAL_MMM02_PARAMS = {'bb_window': 30, 'bb_k': 1.5}


def read_last_trade_price(tape_filename):
    last_price = None
    with open(tape_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 3:
                last_price = float(row[2])
    return last_price


def read_mm_trades(blotter_filename, trader_id='M00'):
    """
    Read the market-maker's own trades from the blotter and infer side from the counterparty id.
    In this BSE setup:
      - counterparty starting with 'S' means the market maker bought
      - counterparty starting with 'B' means the market maker sold
    """
    mm_trades = []

    with open(blotter_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row = [cell.strip() for cell in row]
            if len(row) < 6:
                continue
            if row[0] != trader_id:
                continue
            if row[1] not in ('Trade', 'Trd'):
                continue

            exec_time = float(row[2])
            exec_price = float(row[3])
            party1 = row[4]
            party2 = row[5]

            if party1 == trader_id:
                counterparty = party2
            elif party2 == trader_id:
                counterparty = party1
            else:
                continue

            if counterparty.startswith('S'):
                side = 'buy'
            elif counterparty.startswith('B'):
                side = 'sell'
            else:
                continue

            mm_trades.append({
                'time': exec_time,
                'price': exec_price,
                'side': side,
            })

    mm_trades.sort(key=lambda x: x['time'])
    return mm_trades


def build_equity_curve_from_tape_and_blotter(
    tape_filename,
    blotter_filename,
    trader_id='M00',
    initial_cash=500.0,
):
    """
    Rebuild a mark-to-market equity curve using the tape as the time axis and
    the market-maker's own executions from the blotter.
    """
    mm_trades = read_mm_trades(blotter_filename, trader_id=trader_id)
    mm_idx = 0

    cash = float(initial_cash)
    inventory = 0
    records = []

    with open(tape_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row = [cell.strip() for cell in row]
            if len(row) < 3:
                continue
            if row[0] not in ('Trade', 'Trd'):
                continue

            t = float(row[1])
            market_price = float(row[2])

            while mm_idx < len(mm_trades) and mm_trades[mm_idx]['time'] <= t:
                tr = mm_trades[mm_idx]
                if abs(tr['time'] - t) < 1e-9:
                    if tr['side'] == 'buy':
                        cash -= tr['price']
                        inventory += 1
                    elif tr['side'] == 'sell':
                        cash += tr['price']
                        inventory -= 1
                mm_idx += 1

            equity = cash + inventory * market_price
            records.append({
                'time': t,
                'market_price': market_price,
                'cash': cash,
                'inventory': inventory,
                'equity': equity,
            })

    equity_curve = pd.DataFrame(records)
    if equity_curve.empty:
        equity_curve = pd.DataFrame([
            {
                'time': 0.0,
                'market_price': np.nan,
                'cash': float(initial_cash),
                'inventory': 0,
                'equity': float(initial_cash),
            }
        ])

    equity_curve['running_peak'] = equity_curve['equity'].cummax()
    equity_curve['drawdown_abs'] = equity_curve['running_peak'] - equity_curve['equity']
    equity_curve['drawdown_pct'] = np.where(
        equity_curve['running_peak'] > 0,
        equity_curve['drawdown_abs'] / equity_curve['running_peak'],
        0.0,
    )
    return equity_curve


def compute_max_drawdown(equity_curve_df):
    """
    Return absolute and percentage maximum drawdown from an equity curve DataFrame.
    """
    max_dd_abs = float(equity_curve_df['drawdown_abs'].max())
    max_dd_pct = float(equity_curve_df['drawdown_pct'].max())
    return max_dd_abs, max_dd_pct


def resample_equity_curve_to_grid(equity_curve_df, time_grid):
    """
    Resample one run's equity curve onto a common time grid using
    last-observation-carried-forward.
    """
    eq = equity_curve_df.sort_values('time').reset_index(drop=True)

    if eq.empty:
        return pd.DataFrame({
            'time': time_grid,
            'equity': np.nan,
            'drawdown_pct': np.nan,
        })

    sampled_rows = []
    last_equity = float(eq.iloc[0]['equity'])
    last_drawdown_pct = float(eq.iloc[0]['drawdown_pct'])
    j = 0

    for t in time_grid:
        while j < len(eq) and float(eq.iloc[j]['time']) <= t:
            last_equity = float(eq.iloc[j]['equity'])
            last_drawdown_pct = float(eq.iloc[j]['drawdown_pct'])
            j += 1

        sampled_rows.append({
            'time': float(t),
            'equity': last_equity,
            'drawdown_pct': last_drawdown_pct,
        })

    return pd.DataFrame(sampled_rows)



def save_mean_path_subplot(df, market_noise, results_dir, trader_id='M00', initial_cash=500.0, time_step=1800):
    """
    Build and save a two-panel subplot showing mean cumulative PnL and mean drawdown
    paths over time for MMM01 and MMM02 under one market-noise regime.
    """
    df_sub = df[df['market_noise'] == market_noise]
    if df_sub.empty:
        return None

    end_time = int(7.5 * 60 * 60)
    time_grid = np.arange(0, end_time + 1, time_step)

    path_rows = []

    for _, row in df_sub.iterrows():
        tape_filename = row['trial_prefix'] + '_tape.csv'
        blotter_filename = row['trial_prefix'] + '_blotters.csv'

        equity_curve = build_equity_curve_from_tape_and_blotter(
            tape_filename=tape_filename,
            blotter_filename=blotter_filename,
            trader_id=trader_id,
            initial_cash=initial_cash,
        )
        sampled_curve = resample_equity_curve_to_grid(equity_curve, time_grid)
        sampled_curve['strategy'] = row['strategy']
        sampled_curve['seed'] = row['seed']
        path_rows.append(sampled_curve)

    if len(path_rows) == 0:
        return None

    paths_df = pd.concat(path_rows, ignore_index=True)
    summary_df = paths_df.groupby(['strategy', 'time']).agg(
        mean_equity=('equity', 'mean'),
        mean_drawdown_pct=('drawdown_pct', 'mean'),
    ).reset_index()

    subplot_file = os.path.join(results_dir, f'backtest_mean_paths_subplot_{market_noise}.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for strategy_name in ['MMM01', 'MMM02']:
        sub = summary_df[summary_df['strategy'] == strategy_name].sort_values('time')
        x_hours = sub['time'].to_numpy() / 3600.0
        cumulative_pnl = sub['mean_equity'].to_numpy() - float(initial_cash)
        mean_drawdown_pct = sub['mean_drawdown_pct'].to_numpy() * 100.0

        axes[0].plot(x_hours, cumulative_pnl, marker='o', label=strategy_name)
        axes[1].plot(x_hours, mean_drawdown_pct, marker='o', label=strategy_name)

    axes[0].set_title('(a) Mean cumulative PnL')
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('Mean cumulative PnL')

    axes[1].set_title('(b) Mean drawdown (%)')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('Mean drawdown (%)')

    axes[0].legend()
    axes[1].legend()

    plt.suptitle(f'Mean equity and drawdown paths ({market_noise})')
    plt.tight_layout()
    plt.savefig(subplot_file, dpi=200)
    plt.close()

    return subplot_file


def summarise_run(trial_prefix, strategy_name, seed, trader_id='M00'):
    blotter_filename = trial_prefix + '_blotters.csv'
    tape_filename = trial_prefix + '_tape.csv'

    equity_curve = build_equity_curve_from_tape_and_blotter(
        tape_filename=tape_filename,
        blotter_filename=blotter_filename,
        trader_id=trader_id,
        initial_cash=500.0,
    )
    max_dd_abs, max_dd_pct = compute_max_drawdown(equity_curve)

    buy_x, buy_y, sell_x, sell_y = extract_mm_trades_from_blotter(blotter_filename, trader_id)

    n_buys = len(buy_y)
    n_sells = len(sell_y)
    n_roundtrips = min(n_buys, n_sells)

    paired_buys = np.array(buy_y[:n_roundtrips], dtype=float)
    paired_sells = np.array(sell_y[:n_roundtrips], dtype=float)
    roundtrip_pnls = paired_sells - paired_buys

    realized_pnl = float(roundtrip_pnls.sum())

    end_inventory = n_buys - n_sells
    if end_inventory < 0:
        end_inventory = 0

    last_trade_price = read_last_trade_price(tape_filename)

    unrealized_pnl = 0.0
    if end_inventory > 0 and n_buys > n_sells and last_trade_price is not None:
        last_buy_price = buy_y[-1]
        unrealized_pnl = last_trade_price - last_buy_price

    total_equity_pnl = realized_pnl + unrealized_pnl

    if n_roundtrips > 0:
        win_rate = float((roundtrip_pnls > 0).mean())
        expectancy = float(roundtrip_pnls.mean())
    else:
        win_rate = np.nan
        expectancy = np.nan

    return {
        'seed': seed,
        'strategy': strategy_name,
        'trial_prefix': trial_prefix,
        'buys': n_buys,
        'sells': n_sells,
        'roundtrips': n_roundtrips,
        'end_inventory': end_inventory,
        'realized_pnl': realized_pnl,
        'unrealized_pnl': unrealized_pnl,
        'total_equity_pnl': total_equity_pnl,
        'max_drawdown_abs': max_dd_abs,
        'max_drawdown_pct': max_dd_pct,
        'last_trade_price': last_trade_price,
        'win_rate': win_rate,
        'expectancy': expectancy,
    }


def run_one(strategy_name, seed, market_noise='wide'):
    trial_id = f'backtest_{strategy_name.lower()}_{market_noise}_seed_{seed:03d}'

    if strategy_name == 'MMM02':
        mm_params = FINAL_MMM02_PARAMS
    else:
        mm_params = None

    trial_prefix = run_market_session_with_offset(
        strategy_name=strategy_name,
        trial_id=trial_id,
        random_seed=seed,
        price_offset_filename=COURSEWORK_OFFSET_FILE,
        market_noise=market_noise,
        mm_params=mm_params,
    )
    row = summarise_run(trial_prefix, strategy_name, seed)
    row['market_noise'] = market_noise
    return row


def build_paired_comparison(df):
    """
    Build a paired MMM01-vs-MMM02 comparison table by market_noise and seed.
    """
    pnl_table = df.pivot(index=['market_noise', 'seed'], columns='strategy', values='total_equity_pnl')
    inv_table = df.pivot(index=['market_noise', 'seed'], columns='strategy', values='end_inventory')
    drawdown_table = df.pivot(index=['market_noise', 'seed'], columns='strategy', values='max_drawdown_pct')
    roundtrip_table = df.pivot(index=['market_noise', 'seed'], columns='strategy', values='roundtrips')
    expectancy_table = df.pivot(index=['market_noise', 'seed'], columns='strategy', values='expectancy')

    paired = pd.DataFrame({
        'market_noise': pnl_table.index.get_level_values('market_noise'),
        'seed': pnl_table.index.get_level_values('seed'),
        'MMM01_pnl': pnl_table['MMM01'],
        'MMM02_pnl': pnl_table['MMM02'],
        'delta': pnl_table['MMM02'] - pnl_table['MMM01'],
        'MMM01_drawdown_pct': drawdown_table['MMM01'],
        'MMM02_drawdown_pct': drawdown_table['MMM02'],
        'delta_drawdown_pct': drawdown_table['MMM02'] - drawdown_table['MMM01'],
        'MMM01_roundtrips': roundtrip_table['MMM01'],
        'MMM02_roundtrips': roundtrip_table['MMM02'],
        'delta_roundtrips': roundtrip_table['MMM02'] - roundtrip_table['MMM01'],
        'MMM01_expectancy': expectancy_table['MMM01'],
        'MMM02_expectancy': expectancy_table['MMM02'],
        'delta_expectancy': expectancy_table['MMM02'] - expectancy_table['MMM01'],
        'MMM01_end_inventory': inv_table['MMM01'],
        'MMM02_end_inventory': inv_table['MMM02'],
    }).reset_index(drop=True)

    return paired






def shapiro_test_for_deltas(deltas):
    """
    Shapiro-Wilk normality test for paired deltas.
    """
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) < 3:
        return {
            'shapiro_w_stat': np.nan,
            'shapiro_p_value': np.nan,
        }

    w_stat, p_value = stats.shapiro(deltas)
    return {
        'shapiro_w_stat': float(w_stat),
        'shapiro_p_value': float(p_value),
    }


def paired_t_test_from_deltas(deltas):
    """
    Two-sided paired t-test on paired deltas using scipy.
    Returns NaN stats when there is insufficient data.
    """
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    n = len(deltas)

    if n < 2:
        return {
            'n_pairs': n,
            't_stat': np.nan,
            'p_value': np.nan,
        }

    t_stat, p_value = stats.ttest_1samp(deltas, popmean=0.0)

    return {
        'n_pairs': n,
        't_stat': float(t_stat),
        'p_value': float(p_value),
    }


def wilcoxon_signed_rank_test_from_deltas(deltas):
    """
    Two-sided Wilcoxon signed-rank test on paired deltas using scipy.
    Zero deltas are dropped.
    Returns NaN stats when there is insufficient data.
    """
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    deltas = deltas[deltas != 0]
    n = len(deltas)

    if n < 2:
        return {
            'n_nonzero_pairs': n,
            'w_stat': np.nan,
            'z_stat': np.nan,
            'p_value': np.nan,
        }

    w_stat, p_value = stats.wilcoxon(deltas)

    return {
        'n_nonzero_pairs': n,
        'w_stat': float(w_stat),
        'z_stat': np.nan,
        'p_value': float(p_value),
    }


def compute_summary_stats(df, paired_df):
    """
    Compute descriptive summary statistics for MMM01 and MMM02,
    plus paired comparison stats and simple significance tests,
    grouped by market_noise.
    """
    strategy_stats = df.groupby(['market_noise', 'strategy']).agg(
        mean_pnl=('total_equity_pnl', 'mean'),
        std_pnl=('total_equity_pnl', 'std'),
        mean_max_drawdown_abs=('max_drawdown_abs', 'mean'),
        std_max_drawdown_abs=('max_drawdown_abs', 'std'),
        mean_max_drawdown_pct=('max_drawdown_pct', 'mean'),
        std_max_drawdown_pct=('max_drawdown_pct', 'std'),
        mean_roundtrips=('roundtrips', 'mean'),
        std_roundtrips=('roundtrips', 'std'),
        mean_end_inventory=('end_inventory', 'mean'),
        end_inventory_positive_ratio=('end_inventory', lambda s: (s > 0).mean()),
        mean_win_rate=('win_rate', 'mean'),
        std_win_rate=('win_rate', 'std'),
        mean_expectancy=('expectancy', 'mean'),
        std_expectancy=('expectancy', 'std'),
    ).reset_index()

    comparison_rows = []
    for market_noise, group in paired_df.groupby('market_noise'):
        mmm02_win_rate = (group['delta'] > 0).mean()
        delta_mean = group['delta'].mean()
        delta_std = group['delta'].std()

        shapiro = shapiro_test_for_deltas(group['delta'])
        t_test = paired_t_test_from_deltas(group['delta'])
        wilcoxon = wilcoxon_signed_rank_test_from_deltas(group['delta'])

        drawdown_shapiro = shapiro_test_for_deltas(group['delta_drawdown_pct'])
        drawdown_t_test = paired_t_test_from_deltas(group['delta_drawdown_pct'])
        drawdown_wilcoxon = wilcoxon_signed_rank_test_from_deltas(group['delta_drawdown_pct'])

        roundtrip_shapiro = shapiro_test_for_deltas(group['delta_roundtrips'])
        roundtrip_t_test = paired_t_test_from_deltas(group['delta_roundtrips'])
        roundtrip_wilcoxon = wilcoxon_signed_rank_test_from_deltas(group['delta_roundtrips'])

        expectancy_shapiro = shapiro_test_for_deltas(group['delta_expectancy'])
        expectancy_t_test = paired_t_test_from_deltas(group['delta_expectancy'])
        expectancy_wilcoxon = wilcoxon_signed_rank_test_from_deltas(group['delta_expectancy'])

        comparison_rows.append({
            'market_noise': market_noise,
            'MMM02_win_rate': mmm02_win_rate,
            'delta_mean': delta_mean,
            'delta_std': delta_std,
            'shapiro_w_stat': shapiro['shapiro_w_stat'],
            'shapiro_p_value': shapiro['shapiro_p_value'],
            'paired_t_n': t_test['n_pairs'],
            'paired_t_stat': t_test['t_stat'],
            'paired_t_p_value': t_test['p_value'],
            'wilcoxon_n': wilcoxon['n_nonzero_pairs'],
            'wilcoxon_w_stat': wilcoxon['w_stat'],
            'wilcoxon_z_stat': wilcoxon['z_stat'],
            'wilcoxon_p_value': wilcoxon['p_value'],
            'drawdown_delta_mean': group['delta_drawdown_pct'].mean(),
            'drawdown_delta_std': group['delta_drawdown_pct'].std(),
            'drawdown_shapiro_w_stat': drawdown_shapiro['shapiro_w_stat'],
            'drawdown_shapiro_p_value': drawdown_shapiro['shapiro_p_value'],
            'drawdown_paired_t_n': drawdown_t_test['n_pairs'],
            'drawdown_paired_t_stat': drawdown_t_test['t_stat'],
            'drawdown_paired_t_p_value': drawdown_t_test['p_value'],
            'drawdown_wilcoxon_n': drawdown_wilcoxon['n_nonzero_pairs'],
            'drawdown_wilcoxon_w_stat': drawdown_wilcoxon['w_stat'],
            'drawdown_wilcoxon_z_stat': drawdown_wilcoxon['z_stat'],
            'drawdown_wilcoxon_p_value': drawdown_wilcoxon['p_value'],
            'roundtrip_delta_mean': group['delta_roundtrips'].mean(),
            'roundtrip_delta_std': group['delta_roundtrips'].std(),
            'roundtrip_shapiro_w_stat': roundtrip_shapiro['shapiro_w_stat'],
            'roundtrip_shapiro_p_value': roundtrip_shapiro['shapiro_p_value'],
            'roundtrip_paired_t_n': roundtrip_t_test['n_pairs'],
            'roundtrip_paired_t_stat': roundtrip_t_test['t_stat'],
            'roundtrip_paired_t_p_value': roundtrip_t_test['p_value'],
            'roundtrip_wilcoxon_n': roundtrip_wilcoxon['n_nonzero_pairs'],
            'roundtrip_wilcoxon_w_stat': roundtrip_wilcoxon['w_stat'],
            'roundtrip_wilcoxon_z_stat': roundtrip_wilcoxon['z_stat'],
            'roundtrip_wilcoxon_p_value': roundtrip_wilcoxon['p_value'],
            'expectancy_delta_mean': group['delta_expectancy'].mean(),
            'expectancy_delta_std': group['delta_expectancy'].std(),
            'expectancy_shapiro_w_stat': expectancy_shapiro['shapiro_w_stat'],
            'expectancy_shapiro_p_value': expectancy_shapiro['shapiro_p_value'],
            'expectancy_paired_t_n': expectancy_t_test['n_pairs'],
            'expectancy_paired_t_stat': expectancy_t_test['t_stat'],
            'expectancy_paired_t_p_value': expectancy_t_test['p_value'],
            'expectancy_wilcoxon_n': expectancy_wilcoxon['n_nonzero_pairs'],
            'expectancy_wilcoxon_w_stat': expectancy_wilcoxon['w_stat'],
            'expectancy_wilcoxon_z_stat': expectancy_wilcoxon['z_stat'],
            'expectancy_wilcoxon_p_value': expectancy_wilcoxon['p_value'],
        })

    comparison_stats = pd.DataFrame(comparison_rows)
    return strategy_stats, comparison_stats



def save_analysis_outputs(df, paired_df, strategy_stats, comparison_stats):
    """
    Save paired tables, stats tables, and robustness plots into the results directory.
    """
    paired_file = os.path.join(RESULTS_DIR, 'backtest_paired_comparison.csv')
    strategy_stats_file = os.path.join(RESULTS_DIR, 'backtest_strategy_stats.csv')
    comparison_stats_file = os.path.join(RESULTS_DIR, 'backtest_comparison_stats.csv')

    paired_df.to_csv(paired_file, index=False)
    strategy_stats.to_csv(strategy_stats_file, index=False)
    comparison_stats.to_csv(comparison_stats_file, index=False)

    saved_plot_files = []

    for market_noise in ['wide', 'medium', 'narrow']:
        df_sub = df[df['market_noise'] == market_noise]
        paired_sub = paired_df[paired_df['market_noise'] == market_noise]
        if df_sub.empty or paired_sub.empty:
            continue

        mmm01_pnl = df_sub.loc[df_sub['strategy'] == 'MMM01', 'total_equity_pnl']
        mmm02_pnl = df_sub.loc[df_sub['strategy'] == 'MMM02', 'total_equity_pnl']

        boxplot_file = os.path.join(RESULTS_DIR, f'backtest_pnl_boxplot_{market_noise}.png')
        hist_file = os.path.join(RESULTS_DIR, f'backtest_delta_histogram_{market_noise}.png')
        qqplot_file = os.path.join(RESULTS_DIR, f'backtest_delta_qqplot_{market_noise}.png')
        violin_file = os.path.join(RESULTS_DIR, f'backtest_pnl_violinplot_{market_noise}.png')
        barplot_file = os.path.join(RESULTS_DIR, f'backtest_pnl_barplot_se_{market_noise}.png')

        subplot_file = save_mean_path_subplot(df, market_noise, RESULTS_DIR)

        plt.figure(figsize=(8, 5))
        plt.boxplot([mmm01_pnl, mmm02_pnl], labels=['MMM01', 'MMM02'])
        plt.ylabel('Total equity PnL')
        plt.title(f'Backtest PnL by strategy ({market_noise})')
        plt.tight_layout()
        plt.savefig(boxplot_file, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(paired_sub['delta'], bins=10)
        plt.xlabel('PnL delta (MMM02 - MMM01)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of paired PnL deltas ({market_noise})')
        plt.tight_layout()
        plt.savefig(hist_file, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        stats.probplot(paired_sub['delta'], dist='norm', plot=plt)
        plt.title(f'QQ plot of paired PnL deltas ({market_noise})')
        plt.tight_layout()
        plt.savefig(qqplot_file, dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        data_for_violin = [mmm01_pnl.values, mmm02_pnl.values]
        plt.violinplot(data_for_violin, showmeans=True, showmedians=True)
        plt.xticks([1, 2], ['MMM01', 'MMM02'])
        plt.ylabel('Total equity PnL')
        plt.title(f'Backtest PnL by strategy (violin plot, {market_noise})')
        plt.tight_layout()
        plt.savefig(violin_file, dpi=200)
        plt.close()

        pnl_summary = df_sub.groupby('strategy')['total_equity_pnl'].agg(['mean', 'std', 'count'])
        means = pnl_summary['mean'].values
        ses = (pnl_summary['std'] / np.sqrt(pnl_summary['count'])).values
        labels = pnl_summary.index.tolist()

        plt.figure(figsize=(8, 5))
        x = np.arange(len(labels))
        plt.bar(x, means, yerr=ses, capsize=6)
        plt.xticks(x, labels)
        plt.ylabel('Mean total equity PnL')
        plt.title(f'Mean total equity PnL ± standard error ({market_noise})')
        plt.tight_layout()
        plt.savefig(barplot_file, dpi=200)
        plt.close()

        saved_plot_files.extend([boxplot_file, hist_file, qqplot_file, violin_file, barplot_file])
        if subplot_file is not None:
            saved_plot_files.append(subplot_file)

    return {
        'paired_file': paired_file,
        'strategy_stats_file': strategy_stats_file,
        'comparison_stats_file': comparison_stats_file,
        'plot_files': saved_plot_files,
    }


def run_backtest(seeds, strategies=('MMM01', 'MMM02'), market_noises=('wide', 'medium', 'narrow')):
    results = []

    for market_noise in market_noises:
        for seed in seeds:
            for strategy_name in strategies:
                print(f'Running {strategy_name} with seed={seed}, market_noise={market_noise}, offset={os.path.basename(COURSEWORK_OFFSET_FILE)} ...')
                row = run_one(strategy_name, seed, market_noise=market_noise)
                results.append(row)

    df = pd.DataFrame(results)

    output_file = os.path.join(RESULTS_DIR, 'backtest_summary.csv')
    df.to_csv(output_file, index=False)

    paired_df = build_paired_comparison(df)
    strategy_stats, comparison_stats = compute_summary_stats(df, paired_df)
    saved_files = save_analysis_outputs(df, paired_df, strategy_stats, comparison_stats)

    print('\nSaved backtest summary to:')
    print(output_file)

    print('\nSaved paired comparison to:')
    print(saved_files['paired_file'])

    print('\nSaved strategy stats to:')
    print(saved_files['strategy_stats_file'])

    print('\nSaved comparison stats to:')
    print(saved_files['comparison_stats_file'])

    print('\nSaved plots to:')
    for plot_file in saved_files['plot_files']:
        print(plot_file)

    print('\nPaired comparison:')
    print(paired_df[['market_noise', 'seed', 'MMM01_pnl', 'MMM02_pnl', 'delta']])

    print('\nStrategy statistics:')
    print(strategy_stats)

    print('\nComparison statistics:')
    print(comparison_stats)

    print('\nShapiro-Wilk normality test for paired deltas:')
    print(comparison_stats[['market_noise', 'shapiro_w_stat', 'shapiro_p_value']])

    print('\nPaired t-test summary:')
    print(comparison_stats[['market_noise', 'paired_t_n', 'paired_t_stat', 'paired_t_p_value']])

    print('\nWilcoxon signed-rank summary:')
    print(comparison_stats[['market_noise', 'wilcoxon_n', 'wilcoxon_w_stat', 'wilcoxon_z_stat', 'wilcoxon_p_value']])

    print('\nDrawdown paired test summary:')
    print(comparison_stats[[
        'market_noise',
        'drawdown_delta_mean',
        'drawdown_shapiro_p_value',
        'drawdown_paired_t_p_value',
        'drawdown_wilcoxon_p_value',
    ]])

    print('\nRoundtrip paired test summary:')
    print(comparison_stats[[
        'market_noise',
        'roundtrip_delta_mean',
        'roundtrip_shapiro_p_value',
        'roundtrip_paired_t_p_value',
        'roundtrip_wilcoxon_p_value',
    ]])

    print('\nExpectancy paired test summary:')
    print(comparison_stats[[
        'market_noise',
        'expectancy_delta_mean',
        'expectancy_shapiro_p_value',
        'expectancy_paired_t_p_value',
        'expectancy_wilcoxon_p_value',
    ]])

    return df, paired_df, strategy_stats, comparison_stats


if __name__ == '__main__':
    seeds = list(range(1, 31))   # final run: 30 paired trials
    df, paired_df, strategy_stats, comparison_stats = run_backtest(seeds=list(range(1, 31)),market_noises=('wide',),)
    print(df)
    print(paired_df)
    print(strategy_stats)
    print(comparison_stats)