
import pandas as pd
from typing import Dict, List

from doespy.etl.etl_util import expand_factors, escape_tuple_str
from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.loaders import Loader, PlotLoader

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

COLOR_GRAY = '#999999'

class StatTransformer(Transformer):

    groupby_columns: List[str]

    stats: Dict[str, list]

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        groupby_columns = expand_factors(df, self.groupby_columns)

        # Initialize an empty DataFrame to hold the aggregated results
        aggregated_results = pd.DataFrame()

        # Iterate through each key-value pair in self.stats
        for stat_label, stat_values in self.stats.items():
            # Create a column 'stat_key' to identify which key in 'stats' each row belongs to
            stat_key = f"stat_key_{stat_label}"
            df.loc[df['stat'].isin(stat_values), stat_key] = stat_label

            # Filter out rows that do not match any stat key
            df_filtered = df[df[stat_key].notna()]

            # Group by 'groupby_columns' and 'stat_key', then sum 'stat_value'
            grouped = df_filtered.groupby(groupby_columns + [stat_key])['stat_value'].sum().reset_index()

            # Pivot the table to have separate columns for each stat_key
            pivot_table = grouped.pivot_table(index=groupby_columns, columns=stat_key, values='stat_value', fill_value=0)

            # Merge into the aggregated results
            if aggregated_results.empty:
                aggregated_results = pivot_table
            else:
                aggregated_results = aggregated_results.merge(pivot_table, on=groupby_columns, how='outer')

        # Reset index to turn groupby_columns back into columns
        result = aggregated_results.reset_index()
        return result

    def extract_mpspdz_global_values(self, df):
        # extract a row from this df that contains some global values from mpspdz, for these columns
        # {'player_number': -1, 'player_data_sent': 439.184, 'player_round_number': 2938, 'global_data_sent': 878.368}
        # find the row where these columns are not NaN

        # Additional columns to include
        additional_cols = ['player_number', 'player_data_sent', 'player_round_number', 'global_data_sent']

        # Function to check if only one row is returned per group
        def check_single_row_per_group(group):
            single_row = group[group[additional_cols].notna().all(axis=1)]
            if len(single_row) != 1:
                print(single_row)
                raise ValueError(f"More than one row found for group {group.name}. "
                                 f"This is probably because you have selected more than one run which is likely not what you want. "
                                 f"Consider adapting groupby_columns.")
            return single_row.iloc[0]

        # Select a single row per groupby_columns. Change the selection logic as needed.
        selected_rows = df.groupby(self.groupby_columns).apply(check_single_row_per_group)[additional_cols].reset_index()

        return selected_rows

class TwoDimensionalScatterPlotLoader(PlotLoader):

    plot_cols: List[str]
    plot_cols_values: Dict[str, List[str]]

    symbol_cols: List[str]
    symbol_cols_values: Dict[str, List[str]]

    color_cols: List[str]
    color_cols_values: Dict[str, List[str]]

    y_col: str
    x_col: str
    annotation_col: str

    symbols = ['o', 'v']
    colors = []

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:
        if df.empty:
            return

        output_dir = self.get_output_dir(etl_info)

        n_rows_intial = len(df)
        plot_cols = [(col, self.plot_cols_values[col]) for col in self.plot_cols]
        row_cols = [(col, self.color_cols_values[col]) for col in self.color_cols]
        col_cols = [(col, self.symbol_cols_values[col]) for col in self.symbol_cols]
        for col, allowed in plot_cols + row_cols + col_cols:

            # convert column to string for filtering
            df[col] = df[col].astype(str)

            print(f"Filtering {col} to {allowed}    all={df[col].unique()}")
            # filter out non-relevant results
            df = df[df[col].isin(allowed)]
            # convert to categorical
            df[col] = pd.Categorical(df[col], ordered=True, categories=allowed)

        df.sort_values(by=self.plot_cols + self.color_cols + self.symbol_cols, inplace=True)
        print(f"Filtered out {n_rows_intial - len(df)} rows (based on plot_cols, row_cols, col_cols)  remaining: {len(df)}")

        for idx, df_plot in df.groupby(self.plot_cols):
            print(f"Creating Workload {idx} plot")

            num_rows = np.prod([len(v) for v in self.color_cols_values.values()])
            # number of columns is cartesian product of dictionary values
            num_cols = np.prod([len(v) for v in self.symbol_cols_values.values()])

            # TODO[hly]: Make figsize a property of this class
            fig_size = [3.441760066417601, 2.38667729342697]
            plt_params = {
                'backend': 'ps',
                'axes.labelsize': 18,
                'legend.fontsize': 12,
                'xtick.labelsize': 16,
                'ytick.labelsize': 16,
                'font.size': 14,
                'figure.figsize': fig_size,
                'font.family': 'Times New Roman',
                'lines.markersize': 8
            }

            plt.rcParams.update(plt_params)
            plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            for index, row in df.iterrows():
                # marker = 'o' if row['Setting'] == 'WAN' else 'v'  # Circle for WAN, square for LAN
                # if row['Type'] == 'Com' and row['Setting'] == 'WAN':
                #     color = COLOR_COM
                # elif row['Type'] == 'Com':
                #     color = COLOR_COM_LAN
                # else:
                #     color = COLOR_HASH
                # #     color = COLOR_COM if row['Type'] == 'Com' else COLOR_HASH  # Blue for Com, red for Hash
                # ax.scatter(row[self.x_col], row[self.y_col], c=color, marker=marker, s=75)
                marker = None
                if len(self.symbol_cols) > 0:
                    symbol_index = self.symbol_cols_values[self.symbol_cols[0]].index(row[self.symbol_cols[0]])
                    marker = self.symbols[symbol_index]

                color = None
                if len(self.color_cols) > 0:
                    color_index = self.color_cols_values[self.color_cols[0]].index(row[self.color_cols[0]])
                    color = self.colors[color_index]
                ax.scatter(row[self.x_col], row[self.y_col], marker=marker, color=color, s=75)

                # if row['Type'] == 'Com' and row['Setting'] == 'LAN':
                #     pass
                # else:
                ax.annotate(row[self.annotation_col], (row[self.x_col], row[self.y_col]), textcoords="offset points", xytext=(0,10), ha='center')


            # Add titles and labels
            # plt.title('Comparison with Related Work')
            ax.set_xlabel('Commitment Size (Bytes)')
            ax.set_ylabel('Verification Time (s)')
            ax.set_yscale('log')
            ax.set_xscale('log')
            ax.grid(True, which="major", ls="--", linewidth=0.5)

            # # Create custom legend handles
            # # Create custom legend handles for marker types
            # circle_marker = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='WAN')
            # square_marker = mlines.Line2D([], [], color='gray', marker='v', linestyle='None', markersize=10, label='LAN')
            #
            # # Create custom legend handles for colors
            # com_color = mlines.Line2D([], [], color=COLOR_COM, marker='o', linestyle='None', markersize=10, label='Cryptographic Commitment')
            # hash_color = mlines.Line2D([], [], color=COLOR_HASH, marker='o', linestyle='None', markersize=10, label='Collision-resistant Hash')
            #
            # ax = plt.gca()
            #
            # # Create first legend and add it to the plot
            # first_legend = ax.legend(handles=[circle_marker, square_marker], title='Network', loc='upper left', bbox_to_anchor=(1, 1.02))
            # first_legend._legend_box.align = "left"
            # ax.add_artist(first_legend)

            if len(self.symbol_cols) > 0:
                symbol_lines = []
                for i in range(len(self.symbol_cols_values[self.symbol_cols[0]])):
                    symbol_lines.append(mlines.Line2D([], [], color='gray', marker=self.symbols[i], linestyle='None', markersize=10, label=f'{self.symbol_cols_values[self.symbol_cols[0]][i]}'))
                symbol_legend = ax.legend(handles=symbol_lines, title='Network', loc='upper left', bbox_to_anchor=(1, 1.02))
                symbol_legend._legend_box.align = "left"
                ax.add_artist(symbol_legend)

            if len(self.color_cols) > 0:
                color_lines = []
                for i in range(len(self.color_cols_values[self.color_cols[0]])):
                    color_lines.append(mlines.Line2D([], [], color=self.colors[i], marker='o', linestyle='None', markersize=10, label=f'{self.color_cols_values[self.color_cols[0]][i]}'))
                color_legend = ax.legend(handles=color_lines, title='Network', loc='upper left', bbox_to_anchor=(1, 1.02))
                color_legend._legend_box.align = "left"
                ax.add_artist(color_legend)

            filename = f"consistency_compare_{escape_tuple_str(idx)}"
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False)