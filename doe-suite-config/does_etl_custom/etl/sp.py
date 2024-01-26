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

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import colors as mcolors
import matplotlib.container as mcontainer
import numpy as np

from typing import Tuple, Literal, Any

COLOR_GRAY = '#999999'

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
                print("SUM!", sum, match)
                x.loc[match_this, 'stat_value'] = sum

            return x



        df = df.groupby(self.run_def_cols).apply(t).reset_index(drop=True)

        return df

class CerebroMultiplierTransformer(Transformer):

    run_def_cols: List[str] = ['suite_name', 'exp_name', 'run']
    timer_value_types: List[str] = ["", "_bw", "_rounds"]

    timer_id_cerebro: str = "95"

    timer_thread_help: int = 36 # we assume cerebro can be perfectly parallelized over 36 threads

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
                    print("Multiplied cerebro timer value by ", n_amount_needed_id_value, " for timer ", timer_id, run_id)

                return x
            return t

        df = df.groupby(self.run_def_cols).apply(t_wrap("input")).reset_index(drop=True)
        df = df.groupby(self.run_def_cols).apply(t_wrap("output")).reset_index(drop=True)
        return df

class ComputationMultiplierTransformer(Transformer):

    run_def_cols: List[str] = ['suite_name', 'exp_name', 'run']
    timer_value_types: List[str] = ["", "_bw", "_rounds"]

    timer_id_computation: str = "1102"

    batches_per_epoch_bs_128: Dict[str, int] = {
        "cifar_alexnet": 391,
        "mnist_full": 469,
        "adult": 204,
    }

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        if 'mpc.script_args.n_batches' not in df.columns:
            return df

        def t(x):
            # check if this group contains an n_batches that is set
            if x['mpc.script_args.n_batches'].isna().any():
                return x

            assert x['mpc.script_args.dataset'].unique().size == 1
            assert x['mpc.script_args.n_batches'].unique().size == 1

            dataset = x['mpc.script_args.dataset'].unique()[0]
            n_batches = int(x['mpc.script_args.n_batches'].unique()[0])
            if n_batches == 0:
                return x
            batch_size = int(x['mpc.script_args.batch_size'].unique()[0])
            batch_size_rel = batch_size / 128
            full_epoch = self.batches_per_epoch_bs_128[dataset]

            multiplier = (full_epoch / n_batches) / batch_size_rel
            print(f"Multiplier: {multiplier} for dataset {dataset} and n_batches {n_batches}")

            for timer_value_type in self.timer_value_types:
                timer_id = f"spdz_timer{timer_value_type}_{self.timer_id_computation}"
                if timer_id not in x['stat'].unique():
                    print(f"Skipping run because it does not contain timer {timer_id}")
                    return x
                timer_value = float(x[x['stat'] == timer_id]['stat_value'].values[0])
                print("Original timer value: ", timer_value, " for timer ", timer_id, " after ", timer_value * multiplier)
                timer_value *= multiplier
                x.loc[x['stat'] == timer_id, 'stat_value'] = timer_value

            return x

        df = df.groupby(self.run_def_cols).apply(t).reset_index(drop=True)
        return df

class Sha3MultiplierTransformer(Transformer):

    timer_hash_prefix: str = "sha3_"

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
                    print(f"Set from {total_time_fixed_old} to {total_time_fixed} with {multiplier} for timer_name: {timer_id} and dataset {dataset} {run}")
            # print(x)
            return x

        # print(df.columns)
        df = df.groupby(['suite_name', 'exp_name', 'run']).apply(tran).reset_index(drop=True)
        # print(df)

        return df



class StatTransformer(Transformer):

    groupby_columns: List[str]

    stats: Dict[str, list]

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        assert 'stat_value' in df.columns, f"stat_value not in df.columns: {df.columns}. This might not be the right transformer class for this dataframe."
        df.loc[:, 'stat_value'] = pd.to_numeric(df['stat_value'], errors='coerce')


        ################################Start Hack
        # TODO [Hidde] The field `consistency_prove_verify_bytes_sent` is currently only avaialble as a per-host output.
        #              However, it should be aggregated over all hosts. This is a hack to do that and to make it appear as if this is already done
        # key_cols = ["suite_name", "suite_id", "exp_name", "run", "rep"]
        # df1 = df[df["stat"] == "consistency_prove_verify_bytes_sent"]
        # df_additional = df1.groupby(key_cols).agg({"stat_value": ["sum", "count"]}).reset_index()
        # df_additional["stat"] = "consistency_prove_verify_bytes_sent_global_bytes"
        # df_additional.columns = ["_".join(v) if v[1] else v[0] for v in df_additional.columns.values]
        # df_additional.reset_index(inplace=True, drop=True)  # Resetting the index
        # player_count = df_additional["stat_value_count"].unique()
        #
        # if len(player_count) != 1:
        #     warnings.warn(f"UNEXPECTED WARNING: More than one player count found: {player_count}")
        #
        # df_additional["stat_value"] = df_additional["stat_value_sum"]
        # df_additional.drop(columns=["stat_value_sum", "stat_value_count"], inplace=True)
        # df2 = df1.drop(columns=["host_idx", "stat", "stat_value", "player_number"]).drop_duplicates(subset=key_cols)
        # df2.reset_index(inplace=True, drop=True)
        # df_additional.set_index(key_cols, inplace=True)
        # df2.set_index(key_cols, inplace=True)
        #
        # df_additional = pd.merge(df_additional, df2, on=key_cols, left_index=False, right_index=False).reset_index()
        #
        # df = pd.concat([df, df_additional], ignore_index=True)

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

        print(df)

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
                'lines.markersize': 8
            }

            plt.rcParams.update(plt_params)
            plt.rc('pdf', fonttype=42)  # IMPORTANT to get rid of Type 3

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            for index, row in df_plot.iterrows():
                print(row)
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
