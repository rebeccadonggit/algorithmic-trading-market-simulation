import os
import csv
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
OUTPUT_PATH = os.path.join(RESULTS_DIR, 'scenario_price_dynamics.png')
REFERENCE_OFFSET_FILE = os.path.join(PROJECT_ROOT, 'data', 'spy_1m_2026-03-03.csv')
REFERENCE_OUTPUT_PATH = os.path.join(RESULTS_DIR, 'reference_offset_path.png')
SUPPLY_DEMAND_OUTPUT_PATH = os.path.join(RESULTS_DIR, 'scenario_supply_demand_curves.png')
ROLLING_WINDOW = 30
REPRESENTATIVE_SEED = 1
REPRESENTATIVE_STRATEGY = 'mmm01'

SCENARIO_FILES = {
    'wide': os.path.join(RESULTS_DIR, f'backtest_{REPRESENTATIVE_STRATEGY}_wide_seed_{REPRESENTATIVE_SEED:03d}_tape.csv'),
    'medium': os.path.join(RESULTS_DIR, f'backtest_{REPRESENTATIVE_STRATEGY}_medium_seed_{REPRESENTATIVE_SEED:03d}_tape.csv'),
    'narrow': os.path.join(RESULTS_DIR, f'backtest_{REPRESENTATIVE_STRATEGY}_narrow_seed_{REPRESENTATIVE_SEED:03d}_tape.csv'),
}


SCENARIO_BOUNDS = {
    'wide': {'supply_low': 100, 'supply_high': 200, 'demand_low': 50, 'demand_high': 150},
    'medium': {'supply_low': 115, 'supply_high': 185, 'demand_low': 65, 'demand_high': 135},
    'narrow': {'supply_low': 125, 'supply_high': 175, 'demand_low': 75, 'demand_high': 125},
}


def load_reference_offset(csv_path: str) -> pd.DataFrame:
    """Load the common reference/offset price path used by all scenarios."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Offset/reference file not found: {csv_path}')

    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Remove timezone if present so plotting stays clean.
    if getattr(df['datetime'].dt, 'tz', None) is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)

    df = df.sort_values('datetime').reset_index(drop=True)
    df['reference_price'] = df['close'].astype(float)
    df['rolling_mean'] = df['reference_price'].rolling(ROLLING_WINDOW).mean()
    return df


def load_tape_prices(tape_path: str) -> pd.DataFrame:
    """Load market trade prices from a BSE tape file."""
    if not os.path.exists(tape_path):
        raise FileNotFoundError(f'Tape file not found: {tape_path}')

    rows = []
    with open(tape_path, 'r', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            row_type = str(row[0]).strip()
            if row_type not in ('Trade', 'Trd'):
                continue
            if len(row) < 3:
                continue
            try:
                rows.append(
                    {
                        'time': float(str(row[1]).strip()),
                        'price': float(str(row[2]).strip()),
                    }
                )
            except ValueError:
                continue

    if not rows:
        raise ValueError(f'No trade-price rows found in tape file: {tape_path}')

    df = pd.DataFrame(rows).sort_values('time').reset_index(drop=True)
    df['rolling_mean'] = df['price'].rolling(ROLLING_WINDOW).mean()
    return df


def plot_reference_offset_path() -> str:
    """Plot the common offset/reference price path shared by wide, medium, and narrow scenarios."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = load_reference_offset(REFERENCE_OFFSET_FILE)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['datetime'], df['reference_price'], linewidth=1.2, label='Reference/offset price')
    ax.plot(df['datetime'], df['rolling_mean'], linewidth=1.5, label=f'Rolling mean ({ROLLING_WINDOW})')
    ax.set_title('Common reference/offset price path used across wide, medium, and narrow scenarios')
    ax.set_xlabel('DateTime')
    ax.set_ylabel('Reference price')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(REFERENCE_OUTPUT_PATH, dpi=200)
    plt.show()
    plt.close(fig)

    return REFERENCE_OUTPUT_PATH


def plot_scenario_supply_demand_curves() -> str:
    """Plot stylised supply and demand price ranges for wide, medium, and narrow scenarios."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for ax, (scenario, bounds) in zip(axes, SCENARIO_BOUNDS.items()):
        supply_x = [0, 1]
        demand_x = [0, 1]

        supply_y = [bounds['supply_low'], bounds['supply_high']]
        demand_y = [bounds['demand_high'], bounds['demand_low']]

        ax.plot(supply_x, supply_y, linewidth=2.0, label='Supply schedule')
        ax.plot(demand_x, demand_y, linewidth=2.0, label='Demand schedule')
        ax.fill_between(supply_x, supply_y[0], supply_y[1], alpha=0.08)
        ax.fill_between(demand_x, demand_y[1], demand_y[0], alpha=0.08)
        ax.set_title(f'{scenario.capitalize()} scenario supply/demand bounds')
        ax.set_ylabel('Price')
        ax.legend(loc='best')

    axes[-1].set_xlabel('Stylised quantity scale')
    fig.suptitle('Stylised supply and demand curve ranges under wide, medium, and narrow scenarios')
    fig.tight_layout()
    fig.savefig(SUPPLY_DEMAND_OUTPUT_PATH, dpi=200)
    plt.show()
    plt.close(fig)

    return SUPPLY_DEMAND_OUTPUT_PATH


def plot_scenario_price_dynamics() -> str:
    """Plot representative trade-price dynamics for wide, medium, and narrow scenarios."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, (scenario, filepath) in zip(axes, SCENARIO_FILES.items()):
        df = load_tape_prices(filepath)

        ax.scatter(
            df['time'],
            df['price'],
            s=12,
            alpha=0.45,
            label='Trade price',
        )
        ax.plot(
            df['time'],
            df['rolling_mean'],
            linewidth=1.5,
            label=f'Rolling mean ({ROLLING_WINDOW})',
        )
        ax.set_title(f'{scenario.capitalize()} market-noise scenario')
        ax.set_ylabel('Trade price')
        ax.legend(loc='best')

    axes[-1].set_xlabel('Time')
    fig.suptitle('Representative price dynamics under wide, medium, and narrow scenarios')
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.show()
    plt.close(fig)

    return OUTPUT_PATH


if __name__ == '__main__':
    reference_path = plot_reference_offset_path()
    print('Saved reference/offset path plot to:')
    print(reference_path)

    # supply_demand_path = plot_scenario_supply_demand_curves()
    # print('Saved scenario supply/demand plot to:')
    # print(supply_demand_path)

    saved_path = plot_scenario_price_dynamics()
    print('Saved scenario price-dynamics plot to:')
    print(saved_path)