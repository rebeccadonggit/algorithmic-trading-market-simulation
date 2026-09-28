import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import random

from src.bse import market_session, offset_from_file


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
DEFAULT_DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'spy_1m_2026-03-03.csv')
DEFAULT_STRATEGY = 'MMM02'
DEFAULT_TRIAL_ID = 'default_mmm02_trial'
TRADER_ID = 'M00'
DEFAULT_RANDOM_SEED = 42
FINAL_MMM02_PARAMS = {'bb_window': 30, 'bb_k': 1.5}
# FINAL_MIN_PROFIT_THRESHOLD = 10
# FINAL_MMM02T_PARAMS = {
#     'bb_window': 30,
#     'bb_k': 1.5,
#     'min_profit_threshold': FINAL_MIN_PROFIT_THRESHOLD,
# }


def ensure_results_dir(results_dir=RESULTS_DIR):
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def set_random_seed(seed=DEFAULT_RANDOM_SEED):
    """
    Set the Python and NumPy random seeds so MMM01 and MMM02 can be run
    under the same randomness.
    """
    random.seed(seed)
    np.random.seed(seed)
    return seed


def build_offset_file(source_csv, results_dir=RESULTS_DIR):
    """
    Read the raw market data CSV, keep only time and close, and write a two-column
    offset file into the results directory.
    Returns the generated offset CSV path.
    """
    import pandas as pd

    results_dir = ensure_results_dir(results_dir)
    if not os.path.exists(source_csv):
        raise FileNotFoundError(
            f'Market data not found: {source_csv}. '
            'Place a licensed SPY 1-minute CSV in data/ or pass price_offset_filename explicitly.'
        )
    offset_df = pd.read_csv(source_csv)
    offset_df['datetime'] = pd.to_datetime(offset_df['datetime'])
    offset_df = pd.DataFrame({
        'time': offset_df['datetime'].dt.strftime('%H:%M:%S'),
        'price': offset_df['close']
    })

    offset_file = os.path.join(results_dir, 'offset_file_raw.csv')
    offset_df.to_csv(offset_file, index=False, header=False)
    return offset_file


def run_market_session_with_offset(
    strategy_name=DEFAULT_STRATEGY,
    price_offset_filename=None,
    results_dir=RESULTS_DIR,
    trial_id=None,
    random_seed=DEFAULT_RANDOM_SEED,
    market_noise='wide',
    mm_params=None,
):
    """
    Run one market session with offset-enabled supply/demand schedules.
    Optional market-maker parameters can be passed via mm_params.
    All BSE output files are written into results_dir.
    Returns the absolute trial prefix path, e.g. /.../results/defaultmmm01_trial
    """
    ensure_results_dir(results_dir)
    set_random_seed(random_seed)

    if trial_id is None:
        trial_id = f'default{strategy_name.lower()}_trial'

    if price_offset_filename is None:
        price_offset_filename = DEFAULT_DATA_FILE

    offset_file_raw = build_offset_file(price_offset_filename, results_dir=results_dir)

    start_time = 0
    end_time = 60 * 60 * 7.5
    file_offset = offset_from_file(offset_file_raw, 0, 1, 50, end_time)

    sellers_spec = [('ZIP', 3), ('SHVR', 3), ('GVWY', 3), ('SNPR', 3)]
    buyers_spec = [('ZIP', 3), ('ZIC', 3), ('SHVR', 3), ('GVWY', 3)]
    if strategy_name in ('MMM02','MMM02T') and mm_params is not None:
        marketmaker_spec = [(strategy_name, 1, mm_params)]
    else:
        marketmaker_spec = [(strategy_name, 1)]
    traders_spec = {
        'sellers': sellers_spec,
        'buyers': buyers_spec,
        'mrktmakers': marketmaker_spec,
    }

    if market_noise == 'wide':
        sup_range = (100, 200, file_offset)
        dem_range = (50, 150, file_offset)
    elif market_noise == 'medium':
        sup_range = (115, 185, file_offset)
        dem_range = (65, 135, file_offset)
    elif market_noise == 'narrow':
        sup_range = (125, 175, file_offset)
        dem_range = (75, 125, file_offset)
    else:
        raise ValueError(f'Unknown market_noise: {market_noise}')

    supply_schedule = [
        {'from': start_time, 'to': end_time, 'ranges': [sup_range], 'stepmode': 'random'}
    ]
    demand_schedule = [
        {'from': start_time, 'to': end_time, 'ranges': [dem_range], 'stepmode': 'random'}
    ]

    order_interval = 30
    order_sched = {
        'sup': supply_schedule,
        'dem': demand_schedule,
        'interval': order_interval,
        'timemode': 'drip-jitter',
    }

    trial_path = os.path.join(results_dir, trial_id)

    dump_flags = {
        'dump_blotters': True,
        'dump_lobs': True,
        'dump_strats': True,
        'dump_avgbals': True,
        'dump_tape': True,
    }

    verbose = False

    market_session(trial_path, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)
    print(f'BSE: Run completed. Output prefix: {trial_path}')
    return trial_path


def extract_mm_trades_from_blotter(blotter_filename, trader_id=TRADER_ID):
    """
    # Read MMM01 executions from the blotter.
    Read market-maker executions from the blotter.

    Trade rows are expected to look like:
        M00, Trade, 759.000, 141, S06, M00, 1
        M00, Trade, 859.120, 167, B03, M00, 1

    Interpretation:
        - if counterparty starts with 'S', then M00 bought
        - if counterparty starts with 'B', then M00 sold
    """
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []

    with open(blotter_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) < 6:
                continue

            row = [cell.strip() for cell in row]

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
                buy_x.append(exec_time)
                buy_y.append(exec_price)
            elif counterparty.startswith('B'):
                sell_x.append(exec_time)
                sell_y.append(exec_price)

    return buy_x, buy_y, sell_x, sell_y


def plot_trades_with_mm_markers(trial_prefix, strategy_name=DEFAULT_STRATEGY, trader_id=TRADER_ID):
    """
    Read tape/blotter files from the results directory and plot workshop-style
    trade price vs time, with market-maker buy/sell markers overlaid.
    """
    tape_filename = trial_prefix + '_tape.csv'
    blotter_filename = trial_prefix + '_blotters.csv'

    x = np.empty(0)
    y = np.empty(0)

    with open(tape_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 3:
                time = float(row[1])
                price = float(row[2])
                x = np.append(x, time)
                y = np.append(y, price)

    buy_x, buy_y, sell_x, sell_y = extract_mm_trades_from_blotter(blotter_filename, trader_id)
    # In this single-unit market-maker setup, trades should alternate Buy, Sell, Buy, Sell...
    # So the i-th buy and i-th sell belong to roundtrip number i+1.
    n_roundtrips = min(len(buy_x), len(sell_x))

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, marker='o', color='gray', s=10, alpha=0.45, label='Trade price')

    if len(buy_x) > 0:
        plt.scatter(
            buy_x,
            buy_y,
            marker='^',
            color='red',
            s=90,
            label=f'{strategy_name} buys',
        )
        for i in range(len(buy_x)):
            if i < n_roundtrips:
                label_txt = f'{i + 1:02d}'
            else:
                label_txt = f'B{i + 1:02d}'
            plt.annotate(
                label_txt,
                (buy_x[i], buy_y[i]),
                textcoords='offset points',
                xytext=(4, 6),
                color='red',
                fontsize=9,
            )

    if len(sell_x) > 0:
        plt.scatter(
            sell_x,
            sell_y,
            marker='v',
            color='green',
            s=90,
            label=f'{strategy_name} sells',
        )
        for i in range(len(sell_x)):
            if i < n_roundtrips:
                label_txt = f'{i + 1:02d}'
            else:
                label_txt = f'S{i + 1:02d}'
            plt.annotate(
                label_txt,
                (sell_x[i], sell_y[i]),
                textcoords='offset points',
                xytext=(4, -12),
                color='green',
                fontsize=9,
            )

    plt.xlabel('Time')
    plt.ylabel('Trade Price')
    plt.title(f'Trade price vs time with {strategy_name} executions')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    strategy_name = 'MMM02'
    random_seed = DEFAULT_RANDOM_SEED
    market_noise = 'wide'

    if strategy_name == 'MMM02':
        mm_params = FINAL_MMM02_PARAMS

    else:
        mm_params = None

    trial_prefix = run_market_session_with_offset(
        strategy_name=strategy_name,
        random_seed=random_seed,
        market_noise=market_noise,
        mm_params=mm_params,
    )
    plot_trades_with_mm_markers(trial_prefix, strategy_name=strategy_name)