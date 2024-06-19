import typing
from math import ceil
from enum import Enum
import warnings
import itertools

import pandas as pd
import os
from typing import Dict, List, Union, Optional

from doespy.design.etl_design import MyETLBaseModel
from doespy.etl.etl_util import expand_factors, escape_tuple_str
from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.loaders import Loader, PlotLoader

from does_etl_custom.etl.config import setup_plt
from does_etl_custom.etl.bar_plot_loader import MetricConfig

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
import matplotlib.container as mcontainer
import numpy as np

from typing import Tuple, Literal, Any

COLOR_GRAY = '#999999'

import logging
logging.getLogger('matplotlib.font_manager').disabled = True

class MPCTypeFixTransformer(Transformer):

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        def t(row):
            if row['mpc.protocol_setup'] == 'semi_honest_3':
                row['mpc_type'] = 'sh'

            return row

        df = df.apply(t, axis=1)
        return df

class TimerBandwidthAggregator(Transformer):

    run_def_cols: List[str] = ['suite_name', 'exp_name', 'run']

    verbose: bool = False

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        stat_regex = r"(.*)(_?)spdz_timer_bw_(\d*)"

        def t(x):
            # for all stats that match the regex, check that we have exactly 3, sum them up and replace the three stats with the sum

            # find rows that match stat_regex
            stat_matches = x['stat'].str.match(stat_regex)
            # count how many True we have in stat_matches
            # print(x.columns)
            unique = x[stat_matches]['stat'].unique()
            n_parties = x['mpc.script_args.n_input_parties'].unique()
            assert len(n_parties) == 1
            n_parties = int(n_parties[0])
            for match in unique:
                match_this = x['stat'] == match
                n_true = match_this.sum()
                # check we have exactly 3
                assert n_true == n_parties, f"Found {n_true} instead of {n_parties} that match the regex for {x} ({match})"
                sum = x[match_this]['stat_value'].sum()
                sum = sum * 1000 * 1000 # convert to bytes
                if self.verbose:
                    print("SUM!", sum, match)
                x.loc[match_this, 'stat_value'] = sum

            return x



        df = df.groupby(self.run_def_cols, group_keys=False).apply(t).reset_index(drop=True)

        return df

class CerebroMultiplierTransformer(Transformer):

    run_def_cols: List[str] = ['suite_name', 'exp_name', 'run']
    timer_value_types: List[str] = ["", "_bw", "_rounds"]

    timer_id_cerebro: str = "95"

    timer_thread_help: int = 36 # we assume cerebro can be perfectly parallelized over 36 threads

    verbose: bool = False

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        def t_wrap(type):
            def t(x):
                # check if this group contains all 0s for n_batches
                for timer_value_type in self.timer_value_types:
                    timer_id = f"cerebro_{type}_spdz_timer{timer_value_type}_{self.timer_id_cerebro}"
                    if timer_id not in x['stat'].unique():
                        # print(f"Skipping run because it does not contain timer {timer_id}")
                        return x
                    n_amount_needed_id = f"n_{type}_cerebro"
                    if len(x[x['stat'] == n_amount_needed_id]['stat_value'].unique()) != 1:
                        print(f"Error: n_amount_needed_id is not unique for this group. Aborting")
                        print(x)
                        print(x[x['stat'] == n_amount_needed_id]['stat_value'])
                        exit(1)
                    n_amount_needed_id_value = int(x[x['stat'] == n_amount_needed_id]['stat_value'].values[0])

                    timer_value = float(x[x['stat'] == timer_id]['stat_value'].values[0])
                    if timer_value_type == "" and type == "output":
                        n_amount_needed_id_value = ceil(n_amount_needed_id_value / self.timer_thread_help)
                    timer_value *= n_amount_needed_id_value

                    x.loc[x['stat'] == timer_id, 'stat_value'] = timer_value

                    run_id = x['run'].unique()
                    if self.verbose:
                        print("Multiplied cerebro timer value by ", n_amount_needed_id_value, " for timer ", timer_id, run_id)

                return x
            return t

        df = df.groupby(self.run_def_cols, group_keys=False).apply(t_wrap("input")).reset_index(drop=True)
        df = df.groupby(self.run_def_cols, group_keys=False).apply(t_wrap("output")).reset_index(drop=True)
        return df

class ComputationMultiplierTransformer(Transformer):

    run_def_cols: List[str] = ['suite_name', 'exp_name', 'run']
    timer_value_types: List[str] = ["", "_bw", "_rounds"]

    timer_id_computation: str = "1102"

    batches_per_epoch_bs_128: Dict[str, float] = {
        "cifar_alexnet": 391,
        "mnist_full": 469,
        "adult": 204,
        "glue_qnli_bert": 19.53, # for single epoch, 2500 / 128
    }

    n_epochs: Dict[str, int] = {}

    verbose: bool = False

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        if 'mpc.script_args.n_batches' not in df.columns:
            return df

        def t(x):
            # check if this group contains an n_batches that is set
            if self.n_epochs == False and x['mpc.script_args.n_batches'].isna().any():
                return x

            assert x['mpc.script_args.dataset'].unique().size == 1
            assert x['mpc.script_args.n_batches'].unique().size == 1

            dataset = x['mpc.script_args.dataset'].unique()[0]
            n_batches = int(x['mpc.script_args.n_batches'].unique()[0]) if not x['mpc.script_args.n_batches'].isna().all() else self.batches_per_epoch_bs_128[dataset]
            # if n_batches == 0:
            #     return x
            batch_size = int(x['mpc.script_args.batch_size'].unique()[0])
            batch_size_rel = batch_size / 128
            full_epoch = self.batches_per_epoch_bs_128[dataset]
            n_epochs = self.n_epochs.get(dataset, 1)

            multiplier = (n_epochs * full_epoch / n_batches) / batch_size_rel
            if self.verbose:
                print(f"Multiplier: {multiplier} for dataset {dataset}, n_batches {n_batches} and {n_epochs}")

            for timer_value_type in self.timer_value_types:
                timer_id = f"spdz_timer{timer_value_type}_{self.timer_id_computation}"
                if timer_id not in x['stat'].unique():
                    if self.verbose:
                        print(f"Skipping run because it does not contain timer {timer_id}")
                    return x
                timer_value = float(x[x['stat'] == timer_id]['stat_value'].values[0])
                if self.verbose:
                    print("Original timer value: ", timer_value, " for timer ", timer_id, " after ", timer_value * multiplier)
                timer_value *= multiplier
                x.loc[x['stat'] == timer_id, 'stat_value'] = timer_value

            return x

        df = df.groupby(self.run_def_cols, group_keys=False).apply(t).reset_index(drop=True)
        return df

class Sha3MultiplierTransformer(Transformer):

    timer_hash_prefix: str = "sha3_"

    verbose: bool = False

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        timers = [
            { "total": "98", "variables": [ "90", "91"], "prefix": "input" },
            { "total": "97", "variables": [ "93", "94"], "prefix": "output" }
        ]

        # timer_ids_variable = [
        #     "90", # TIMER_INPUT_CONSISTENCY_SHA_BIT_DECOMPOSE
        #     "91", # TIMER_INPUT_CONSISTENCY_SHA_HASH_VARIABLE
        #
        #     "93",
        #     "94"]
        # timer_ids_fixed = [
        #     "92", # TIMER_INPUT_CONSISTENCY_SHA_HASH_FIXED
        # ]
        # timer_id_input_consistency = "98" # TIMER_INPUT_CONSISTENCY_CHECK

        def tran(x):
            # access stat_value for stat spdz_timer_98
            for timer in timers:
                timer_id_total = timer["total"]
                timer_ids_variable = timer["variables"]
                timer_prefix = f'{timer["prefix"]}_'

                for timer_value_type in ["", "_bw", "_rounds"]:
                    timer_id = f"{self.timer_hash_prefix}{timer_prefix}spdz_timer{timer_value_type}_{timer_id_total}"
                    if x[x['stat'] == timer_id].empty:
                        # print("Timer ID is empty ", timer_id)
                        continue
                    if self.verbose:
                        print("Running for timer_value_type: ", timer_value_type)
                    total_time_fixed_old = x[x['stat'] == timer_id]['stat_value'].values[0]
                    total_time_fixed = total_time_fixed_old
                    # print("Total time fixed: ", total_time_fixed)
                    total_time_var = 0
                    for var in timer_ids_variable:
                        timer_id_var = f"{self.timer_hash_prefix}{timer_prefix}spdz_timer{timer_value_type}_{var}"
                        if x[x['stat'] == timer_id_var].empty:
                            continue
                        var_time = float(x[x['stat'] == timer_id_var]['stat_value'].values[0])
                        total_time_fixed -= var_time
                        total_time_var += var_time

                    if total_time_var == 0:
                        # print("Skipping run because total_time_var is 0")
                        return x
                    # print("Total time fixed minus variable part: ", total_time_fixed)
                    multiplier = float(x['mpc.script_args.sha3_approx_factor'].unique()[0])
                    dataset = x['mpc.script_args.dataset'].unique()
                    run = x['run'].unique()
                    # print("Multiplier: ", multiplier, total_time_var, dataset)
                    total_time_fixed += total_time_var * multiplier
                    # now set the value for x
                    x.loc[x['stat'] == timer_id, 'stat_value'] = total_time_fixed
                    if self.verbose:
                        print(f"Set from {total_time_fixed_old} to {total_time_fixed} with {multiplier} for timer_name: {timer_id} and dataset {dataset} {run}")
            # print(x)
            return x

        df = df.groupby(['suite_name', 'exp_name', 'run'], group_keys=False).apply(tran).reset_index(drop=True)

        return df



class StatTransformer(Transformer):

    groupby_columns: List[str]

    stats: Dict[str, list]

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        # print rows of df where stat == spdz_timer_1102

        assert 'stat_value' in df.columns, f"stat_value not in df.columns: {df.columns}. This might not be the right transformer class for this dataframe."
        df.loc[:, 'stat_value'] = pd.to_numeric(df['stat_value'], errors='coerce')

        # not sure where this gets interpreted as a float
        df["host_idx"] = df["host_idx"].fillna(0)
        df['host_idx'] = df['host_idx'].astype(int)

        ###############################End Hack



        groupby_columns = expand_factors(df, self.groupby_columns)

        # Initialize an empty DataFrame to hold the aggregated results
        aggregated_results = pd.DataFrame()

        # check that each stat value specified in self.stats is contained in stat
        for stat_label, stat_values in self.stats.items():
            for stat_value in stat_values:
                if stat_value not in df['stat'].unique():
                    raise ValueError(f"Stat value {stat_value} for stat {stat_label} not found in df['stat'].unique()"
                                     f"df['stat'].unique()={df['stat'].unique()}")


        # Iterate through each key-value pair in self.stats
        for stat_label, stat_values in self.stats.items():
            # Create a column 'stat_key' to identify which key in 'stats' each row belongs to
            stat_key = f"stat_key_{stat_label}"
            df.loc[df['stat'].isin(stat_values), stat_key] = stat_label

            # Filter out rows that do not match any stat key
            df_filtered = df[df[stat_key].notna()]


            # Group by 'groupby_columns' and 'stat_key', then ensure we have a single 'stat_value' and select it
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

class GroupByAppendTransformer(Transformer):

    groupby_columns: List[str]

    metrics: Dict[str, list]

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        groupby_columns = expand_factors(df, self.groupby_columns)

        # group by groupby_columns and aggregate metrics. Concatenate the result as columns to the original df
        df_res = df.groupby(groupby_columns).agg(self.metrics).reset_index()
        df = df.merge(df_res, on=groupby_columns, how='outer')

        return df

class AddTransformer(Transformer):

    result_col: str
    add_cols: List[typing.Any]
    divisors: List[int]

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        if len(self.divisors) == 0:
            self.divisors = [1] * len(self.add_cols)

        for i, add_col in enumerate(self.add_cols):
            if isinstance(add_col, list):
                self.add_cols[i] = tuple(add_col)

        df_temp = df[self.add_cols].copy()
        df_temp = df_temp.div(self.divisors)
        df[self.result_col] = df_temp.sum(axis=1)

        return df

class TwoDimensionalScatterPlotLoader(PlotLoader):

    plot_cols: List[str]
    plot_cols_values: Dict[str, List[str]]

    symbol_cols: List[str]
    symbol_cols_values: Dict[str, List[str]]
    symbol_cols_labels: Dict[str, List[str]]
    symbol_cols_title: str

    color_cols: List[str]
    color_cols_values: Dict[str, List[Union[str, None]]]
    color_cols_labels: Dict[str, List[str]]
    color_cols_title: str

    y_col: str
    x_col: str
    annotation_col: str
    annotation_labels: Dict[str, str]

    symbols = ['o', 'v']
    colors = [['royalblue', 'deepskyblue'], ['red', 'lightcoral']]

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
            # df[col] = df[col].fillna('None')
            df[col] = pd.Categorical(df[col], ordered=True, categories=allowed)

        df.sort_values(by=self.plot_cols + self.color_cols + self.symbol_cols, inplace=True)
        print(f"Filtered out {n_rows_intial - len(df)} rows (based on plot_cols, row_cols, col_cols)  remaining: {len(df)}")

        for idx_group, df_plot in df.groupby(self.plot_cols):
            print(f"Creating Workload {idx_group} plot")

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
                'lines.markersize': 8,
                'pdf.fonttype': 42,
                'ps.fonttype': 42
            }

            plt.rcParams.update(plt_params)
            plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            for index, row in df_plot.iterrows():
                marker = None
                symbol_index = 0
                if len(self.symbol_cols) > 0:
                    symbol_index = self.symbol_cols_values[self.symbol_cols[0]].index(row[self.symbol_cols[0]])
                    marker = self.symbols[symbol_index]

                color = None
                if len(self.color_cols) > 0:
                    color_index = self.color_cols_values[self.color_cols[0]].index(row[self.color_cols[0]])
                    color = self.colors[color_index][symbol_index]
                ax.scatter(row[self.x_col], row[self.y_col], marker=marker, color=color, s=75)

                # if row['Type'] == 'Com' and row['Setting'] == 'LAN':
                #     pass
                # else:
                ax.annotate(self.annotation_labels[row[self.annotation_col]], (row[self.x_col], row[self.y_col]), textcoords="offset points", xytext=(0,10), ha='center')


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
                    symbol_lines.append(mlines.Line2D([], [], color='gray', marker=self.symbols[i], linestyle='None', markersize=10, label=f'{self.symbol_cols_labels[self.symbol_cols[0]][i]}'))
                symbol_legend = ax.legend(handles=symbol_lines, title=self.symbol_cols_title, loc='upper left', bbox_to_anchor=(1, 1.02))
                symbol_legend._legend_box.align = "left"
                ax.add_artist(symbol_legend)

            if len(self.color_cols) > 0:
                color_lines = []
                for i in range(len(self.color_cols_values[self.color_cols[0]])):
                    color_lines.append(mlines.Line2D([], [], color=self.colors[i][0], marker='o', linestyle='None', markersize=10, label=f'{self.color_cols_labels[self.color_cols[0]][i]}'))
                color_legend = ax.legend(handles=color_lines, title=self.color_cols_title, loc='upper left', bbox_to_anchor=(1, .8))
                color_legend._legend_box.align = "left"
                ax.add_artist(color_legend)

            plt.subplots_adjust(right=0.65)

            # Extra margin on the x-axis
            # x_min, x_max = df_plot[self.x_col].min(), df_plot[self.x_col].max()
            # x_range = x_max - x_min
            # plt.xlim([x_min, x_max + 10 * x_range])
            #
            # # Extra margin on the y-axis
            # y_min, y_max = df_plot[self.y_col].astype(float).min(), df_plot[self.y_col].astype(float).max()
            # y_range = y_max - y_min
            # plt.ylim([3.0, y_max + 7000])

            filename = f"consistency_compare_{escape_tuple_str(idx_group)}"
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False)

class CerebroSpecificMultiplierTransformer(Transformer):

    run_def_cols: List[str] = ['suite_name', 'exp_name', 'run']
    timer_value_types: List[str] = ["", "_bw", "_rounds"]

    timer_id_cerebro: str = "95"

    timer_thread_help: int = 36 # we assume cerebro can be perfectly parallelized over 36 threads

    verbose: bool = False

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        def t_wrap(type):
            def t(x):
                # check if this group contains all 0s for n_batches
                for timer_value_type in self.timer_value_types:
                    timer_id = f"cerebro_{type}_spdz_timer{timer_value_type}_{self.timer_id_cerebro}"
                    if timer_id not in x['stat'].unique():
                        # print(f"Skipping run because it does not contain timer {timer_id}")
                        return x
                    n_amount_needed_id = f"n_{type}_cerebro"
                    if len(x[x['stat'] == n_amount_needed_id]['stat_value'].unique()) != 1:
                        print(f"Error: n_amount_needed_id is not unique for this group. Aborting")
                        print(x)
                        print(x[x['stat'] == n_amount_needed_id]['stat_value'])
                        exit(1)
                    n_amount_needed_id_value = int(x[x['stat'] == n_amount_needed_id]['stat_value'].values[0])
                    if n_amount_needed_id_value == 6:
                        n_amount_needed_id_value = 3

                    timer_value = float(x[x['stat'] == timer_id]['stat_value'].values[0])
                    if timer_value_type == "" and type == "output":
                        n_amount_needed_id_value = ceil(n_amount_needed_id_value / self.timer_thread_help)
                    timer_value *= n_amount_needed_id_value

                    x.loc[x['stat'] == timer_id, 'stat_value'] = timer_value

                    run_id = x['run'].unique()
                    if self.verbose:
                        print("Multiplied cerebro timer value by ", n_amount_needed_id_value, " for timer ", timer_id, run_id)

                return x
            return t

        df = df.groupby(self.run_def_cols).apply(t_wrap("input")).reset_index(drop=True)
        # df = df.groupby(self.run_def_cols).apply(t_wrap("output")).reset_index(drop=True)
        return df

class ComputationSpecificMultiplierTransformer(Transformer):

    run_def_cols: List[str] = ['suite_name', 'exp_name', 'run']
    timer_value_types: List[str] = ["mus"]

    timer_name: str = "1102"

    percentage_model_size_training_data_size: Dict[str, float] = {
        "cifar_alexnet": 0.071,
        "mnist_full": 0.026,
        "adult": 0.003,
    }

    verbose: bool = False

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        # if 'mpc.script_args.n_batches' not in df.columns:
        #     return df

        def t(x):
            # check if this group contains an n_batches that is set
            assert x['mpc.script_args.dataset'].unique().size == 1
            assert x['mpc.script_args.consistency_check'].unique().size == 1

            if x['mpc.script_args.consistency_check'].unique()[0] != "cerebro":
                return x

            dataset = x['mpc.script_args.dataset'].unique()[0]
            percentage = self.percentage_model_size_training_data_size[dataset]

            multiplier = (1 - percentage)
            if self.verbose:
                print(f"[SPECIFIC] Multiplier: {multiplier} for dataset {dataset}")

            # consistency_convert_shares_share_switch_input_mus

            for timer_value_type in self.timer_value_types:
                timer_id = f"{self.timer_name}_{timer_value_type}"
                if timer_id not in x['stat'].unique():
                    print(f"Skipping run because it does not contain timer {timer_id}")
                    return x
                timer_values = x[x['stat'] == timer_id]['stat_value'].values
                # Multiply all times_values (float) with the multiplier
                timer_values_upd = [float(timer_value) * multiplier for timer_value in timer_values]
                if self.verbose:
                    print("[SPECIFIC] Original timer value: ", timer_values, " for timer ", timer_id, " after ", timer_values_upd)
                # timer_value *= multiplier
                x.loc[x['stat'] == timer_id, 'stat_value'] = timer_values_upd

            return x

        df = df.groupby(self.run_def_cols).apply(t).reset_index(drop=True)
        return df

class FilteredTableLoader(Loader):

    metrics: Dict[str, MetricConfig]

    plot_cols: List[str]
    group_cols: List[str] # for each combination of these clumns, will have a top-level column

    bar_cols: List[str] # for each value of these columns, will have a bar in each group
    cols_values_filter: Dict[str, List[str]]

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> pd.DataFrame:
        df_filtered = self.filter_df(df)

        output_dir = self.get_output_dir(etl_info)
        filename = f"table_filtered"

        df_filtered = df_filtered[['mpc.script_args.dataset', 'network_type', 'mpc_type', 'mpc_time_s', 'consistency_args.type', 'auditing_overhead_s']]

        # Pivot the DataFrame
        df_pivot = df_filtered.pivot_table(
            index=['mpc.script_args.dataset', 'network_type', 'mpc_type', 'mpc_time_s'],
            columns='consistency_args.type',
            values='auditing_overhead_s',
            aggfunc='first'  # or 'mean' or 'sum' depending on your requirement
        ).reset_index()

        def first_non_nan(series):
            return series.dropna().iloc[0] if not series.dropna().empty else None

        # Rename the columns for clarity
        df_pivot.columns.name = None
        df_pivot = df_pivot.rename(columns={'sha3s': 'auditing_overhead_sha3s', 'cerebro': 'auditing_overhead_cerebro', 'pc': 'auditing_overhead_pc'})
        df_merged = df_pivot.groupby(['mpc.script_args.dataset', 'network_type', 'mpc_type']).agg({
            'mpc_time_s': 'first',
            'auditing_overhead_sha3s': first_non_nan,
            'auditing_overhead_cerebro': first_non_nan,
            'auditing_overhead_pc': first_non_nan
        }).reset_index()
        # print("DF pivot", df_merged)

        latex = self.dataframe_to_latex(df_merged)
        print("LATEX")
        print(latex)

    def dataframe_to_latex(self, df):
        latex_code = r"\begin{table*}[h!]\centering" + "\n"
        latex_code += r"\begin{tabular}{ccccccc}" + "\n"
        latex_code += r"\toprule" + "\n"
        latex_code += r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{Network} & \multirow{2}{*}{MPC} & \multirow{2}{*}{Training Time} & \multicolumn{3}{c}{Consistency Overhead} \\" + "\n"
        latex_code += r"\cmidrule{5-7}" + "\n"
        latex_code += r"& & & & \textbf{Ours} & \gls{a:sha3} & \gls{a:ped} \\" + "\n"
        latex_code += r"\midrule" + "\n"

        last_dataset = ""
        last_network = ""
        last_mpc = ""

        def label(lbl):
            labels = {
                "sha3s": r"\gls{a:sha3}",
                "cerebro": r"\gls{a:ped}",
                "pc": "Ours",

                "cifar_alexnet": "\gls{sc:cifar}",
                "mnist_full": r"\gls{sc:mnist}",
                "adult": r"\gls{sc:adult}",
                "glue_qnli_bert": r"\gls{sc:qnli}",

                "lan": "LAN",
                "wan": "WAN",

                "sh": "SH",
                "mal": "MAL",
            }
            return labels.get(lbl, lbl)

        format_map = {
            ("adult", "lan"): "s",
            ("adult", "wan"): "h",
            ("mnist_full", "lan"): "m",
            ("mnist_full", "wan"): "d",
            ("cifar_alexnet", "lan"): "h",
            ("cifar_alexnet", "wan"): "d",
            ("glue_qnli_bert", "lan"): "h",
            ("glue_qnli_bert", "wan"): "d",
        }

        def ro(num, relative_to=None, time_unit=None):
            if pd.isna(num):
                return ""
            value = self.format_axis_label(num, time_unit)
            if relative_to is None:
                return value

            percentage = num / relative_to
            percentage_str = f" ({percentage:.0f}x)"

            return f"{value}{percentage_str}"

        for index, row in df.iterrows():
            dataset = row['mpc.script_args.dataset']
            network = row['network_type']
            mpc = row['mpc_type']

            if dataset != last_dataset and last_dataset != "":
                latex_code += r"\midrule"

            dataset_cell = f"\multirow{{{len(df[df['mpc.script_args.dataset'] == dataset])}}}{{*}}{{{label(dataset)}}}" if dataset != last_dataset else ""
            network_cell = f"\multirow{{{len(df[(df['mpc.script_args.dataset'] == dataset) & (df['network_type'] == network)])}}}{{*}}{{{label(network)}}}" if network != last_network else ""
            mpc_cell = f"\multirow{{{len(df[(df['mpc.script_args.dataset'] == dataset) & (df['network_type'] == network) & (df['mpc_type'] == mpc)])}}}{{*}}{{{label(mpc)}}}" if mpc != last_mpc else ""

            time_unit = format_map[(dataset, network)]
            latex_code += fr"{dataset_cell} & {network_cell} & {mpc_cell} & {ro(row['mpc_time_s'], None, time_unit)} & {ro(row['auditing_overhead_pc'], None, time_unit)} & {ro(row['auditing_overhead_sha3s'], row['auditing_overhead_pc'], time_unit)} & {ro(row['auditing_overhead_cerebro'], row['auditing_overhead_pc'], time_unit)}  \\" + "\n"

            if dataset != last_dataset:
                last_dataset = dataset
            if network != last_network:
                last_network = network
            if mpc != last_mpc:
                last_mpc = mpc

            latex_code += "\n"

        latex_code += r"\bottomrule" + "\n"
        latex_code += r"\end{tabular}" + "\n"
        latex_code += r"\caption{Overhead of consistency approaches we evaluate relative to (extrapolated) end-to-end training. Multipliers in parentheses are slowdown over ours. Time is given in seconds (s), minutes (m), hours (h), days (d) and weeks (w).}" + "\n"
        latex_code += r"\ltab{e2e_training}" + "\n"
        latex_code += r"\end{table*}" + "\n"

        return latex_code

    def format_axis_label(self, value, time_unit):
        """
        Custom formatting function for y-axis labels.
        """

        def format(value):

            if abs(value) < 0.001:
                formatted_number = f'{value:.4f}'
            elif abs(value) < 0.01:
                formatted_number = f'{value:.3f}'
            elif abs(value) < 0.1:
                formatted_number = f'{value:.2f}'
            else:
                formatted_number = f'{value:.1f}'

            # remove trailing zero
            if "." in formatted_number:
                formatted_number = formatted_number.rstrip('0').rstrip('.')

            return formatted_number

        # val is in seconds
        def format_duration(seconds, unit):
            intervals = (
                ('w', 604800),  # 60 * 60 * 24 * 7
                ('d', 86400),    # 60 * 60 * 24
                ('h', 3600),    # 60 * 60
                ('m', 60),
                ('s', 1),
            )

            for name, count in intervals:
                value = seconds / count
                if value >= 1:
                    return format(value) + name

        formatted_number = format_duration(value, time_unit)
        val = formatted_number

        return val

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:

        n_rows_intial = len(df)
        df_filtered = df.copy()

        plot_cols = [(col, self.cols_values_filter[col]) for col in self.plot_cols]
        group_cols = [(col, self.cols_values_filter[col]) for col in self.group_cols]
        bar_cols = [(col, self.cols_values_filter[col]) for col in self.bar_cols]
        # filter out non-relevant results
        for col, allowed in plot_cols + group_cols + bar_cols:

            # convert column to string for filtering
            try:
                df_filtered[col] = df_filtered[col].astype(str)
            except KeyError:
                raise KeyError(f"col={col} not in df.columns={df.columns}")

            print(f"Filtering {col} to {allowed}    all={df[col].unique()}")
            # filter out non-relevant results
            df_filtered = df_filtered[df_filtered[col].isin(allowed)]
            # convert to categorical
            df_filtered[col] = pd.Categorical(df_filtered[col], ordered=True, categories=allowed)
        df_filtered.sort_values(by=self.plot_cols + self.group_cols + self.bar_cols, inplace=True)

        print(f"Filtered out {n_rows_intial - len(df_filtered)} rows (based on plot_cols, row_cols, col_cols)  remaining: {len(df_filtered)}")

        return df_filtered

    def aggregate_data(self, df_plot, metric_cfg: MetricConfig):

        # allow changing the unit of the metric before calculating the mean / std
        df_plot[metric_cfg.bar_part_cols] = df_plot[metric_cfg.bar_part_cols] * metric_cfg.y_unit_multiplicator / metric_cfg.y_unit_divider

        # check if multiple rows exist for self.group_cols + self.bar_cols
        # if so, output warning
        # create a bar for each bar_cols group in each group_cols group
        grouped_over_reps = df_plot.groupby(by = self.group_cols + self.bar_cols)
        # print first group rows
        for group in grouped_over_reps.groups:
            if len(grouped_over_reps.get_group(group)) > 1:
                # TODO [SOMETHING IS HARDCODED HERE?]
                print(f"Group rows: {grouped_over_reps.get_group(group)}")
                print(f"Const args: {grouped_over_reps.get_group(group)['consistency_args.type']}")


        combined = grouped_over_reps[metric_cfg.bar_part_cols].agg(['mean', 'std'])

        combined[("$total$", "mean")] = combined.loc[:, pd.IndexSlice[metric_cfg.bar_part_cols, "mean"]].sum(axis=1)
        for col in metric_cfg.bar_part_cols:
            combined[(f"$total_share_{col}$", "mean")] = combined[col]["mean"] / combined["$total$"]["mean"]
            combined[(f"$total_factor_{col}$", "mean")] = combined["$total$"]["mean"] / combined[col]["mean"]


        return combined.reset_index()

    def save_data(self, df: pd.DataFrame, filename: str, output_dir: str, output_filetypes: List[str] = ["html"]):
        """:meta private:"""
        os.makedirs(output_dir, exist_ok=True)

        for ext in output_filetypes:
            if ext == "html":
                html_table = df.to_html()
                path = os.path.join(output_dir, f"{filename}.html")

                with open(path, 'w') as file:
                    file.write(html_table)
            else:
                raise ValueError(f"PlotLoader: Unknown file type {ext}")


class ActualDurationLoader(Loader):

    metrics: Dict[str, MetricConfig]

    plot_cols: List[str]
    group_cols: List[str] # for each combination of these clumns, will have a top-level column

    bar_cols: List[str] # for each value of these columns, will have a bar in each group
    cols_values_filter: Dict[str, List[str]]

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> pd.DataFrame:
        df_filtered = self.filter_df(df)

        output_dir = self.get_output_dir(etl_info)
        filename = f"table_filtered"

        # df_filtered = df_filtered[['mpc.script_args.dataset', 'network_type', 'mpc_type', 'total_time_s', 'consistency_args.type']]

        # sum over suite_name
        df_sum = df_filtered.groupby(['suite_name'], group_keys=False).agg({ 'total_time_s': ['sum', 'count'] }).reset_index()

        df_sum['time'] = df_sum['total_time_s']['sum'].apply(lambda x: self.format_duration(x))

        startup_cost_constant = 10 * 60 # 5 minutes to provision and install servers
        end_cost_constant = 1 * 60 # 1 minute to shut down servers
        constant_cost = startup_cost_constant + end_cost_constant
        per_run_provision_cost = 30 * 2 # 30 seconds per run to load results and set new, times two cause compilation
        per_run_compile_time = 150 # 45 seconds on average?
        per_run_cost = per_run_provision_cost + per_run_compile_time
        df_sum['estim_time'] = df_sum \
            .apply(lambda x: self.format_duration(x[('total_time_s', 'sum')] +
                                                  x[('total_time_s', 'count')] * per_run_cost +
                                                  constant_cost), axis=1)

        total_time = df_sum['total_time_s']['sum'].sum()
        total_price = total_time * 3 * 1.746 * 0.93 / 3600
        print("Total price", total_price, "euro")

        # output as markdown
        print(df_sum.to_markdown())

    def format_duration(self, seconds):
        def format(value):

            if abs(value) < 0.001:
                formatted_number = f'{value:.4f}'
            elif abs(value) < 0.01:
                formatted_number = f'{value:.3f}'
            elif abs(value) < 0.1:
                formatted_number = f'{value:.2f}'
            else:
                formatted_number = f'{value:.1f}'

            # remove trailing zero
            if "." in formatted_number:
                formatted_number = formatted_number.rstrip('0').rstrip('.')

            return formatted_number

        intervals = (
            ('w', 604800),  # 60 * 60 * 24 * 7
            ('d', 86400),    # 60 * 60 * 24
            ('h', 3600),    # 60 * 60
            ('m', 60),
            ('s', 1),
        )

        for name, count in intervals:
            value = seconds / count
            if value >= 1:
                return format(value) + name

    def filter_df(self, df: pd.DataFrame) -> pd.DataFrame:

        n_rows_intial = len(df)
        df_filtered = df.copy()

        plot_cols = [(col, self.cols_values_filter[col]) for col in self.plot_cols]
        group_cols = [(col, self.cols_values_filter[col]) for col in self.group_cols]
        bar_cols = [(col, self.cols_values_filter[col]) for col in self.bar_cols]
        # filter out non-relevant results
        for col, allowed in plot_cols + group_cols + bar_cols:

            # convert column to string for filtering
            try:
                df_filtered[col] = df_filtered[col].astype(str)
            except KeyError:
                raise KeyError(f"col={col} not in df.columns={df.columns}")

            print(f"Filtering {col} to {allowed}    all={df[col].unique()}")
            # filter out non-relevant results
            df_filtered = df_filtered[df_filtered[col].isin(allowed)]
            # convert to categorical
            df_filtered[col] = pd.Categorical(df_filtered[col], ordered=True, categories=allowed)
        df_filtered.sort_values(by=self.plot_cols + self.group_cols + self.bar_cols, inplace=True)

        print(f"Filtered out {n_rows_intial - len(df_filtered)} rows (based on plot_cols, row_cols, col_cols)  remaining: {len(df_filtered)}")

        return df_filtered

    def aggregate_data(self, df_plot, metric_cfg: MetricConfig):

        # allow changing the unit of the metric before calculating the mean / std
        df_plot[metric_cfg.bar_part_cols] = df_plot[metric_cfg.bar_part_cols] * metric_cfg.y_unit_multiplicator / metric_cfg.y_unit_divider

        # check if multiple rows exist for self.group_cols + self.bar_cols
        # if so, output warning
        # create a bar for each bar_cols group in each group_cols group
        grouped_over_reps = df_plot.groupby(by = self.group_cols + self.bar_cols)
        # print first group rows
        for group in grouped_over_reps.groups:
            if len(grouped_over_reps.get_group(group)) > 1:
                # TODO [SOMETHING IS HARDCODED HERE?]
                print(f"Group rows: {grouped_over_reps.get_group(group)}")
                print(f"Const args: {grouped_over_reps.get_group(group)['consistency_args.type']}")


        combined = grouped_over_reps[metric_cfg.bar_part_cols].agg(['mean', 'std'])

        combined[("$total$", "mean")] = combined.loc[:, pd.IndexSlice[metric_cfg.bar_part_cols, "mean"]].sum(axis=1)
        for col in metric_cfg.bar_part_cols:
            combined[(f"$total_share_{col}$", "mean")] = combined[col]["mean"] / combined["$total$"]["mean"]
            combined[(f"$total_factor_{col}$", "mean")] = combined["$total$"]["mean"] / combined[col]["mean"]


        return combined.reset_index()

    def save_data(self, df: pd.DataFrame, filename: str, output_dir: str, output_filetypes: List[str] = ["html"]):
        """:meta private:"""
        os.makedirs(output_dir, exist_ok=True)

        for ext in output_filetypes:
            if ext == "html":
                html_table = df.to_html()
                path = os.path.join(output_dir, f"{filename}.html")

                with open(path, 'w') as file:
                    file.write(html_table)
            else:
                raise ValueError(f"PlotLoader: Unknown file type {ext}")
