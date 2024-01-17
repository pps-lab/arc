import typing
from math import ceil
from enum import Enum
import warnings

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

COLOR_GRAY = '#999999'

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
                    timer_value *= n_amount_needed_id_value
                    if timer_value_type == "":
                        timer_value /= self.timer_thread_help
                    x.loc[x['stat'] == timer_id, 'stat_value'] = timer_value

                    run_id = x['run'].unique()
                    print("Multiplied cerebro timer value by ", n_amount_needed_id_value, " for timer ", timer_id, run_id)

                return x
            return t

        df = df.groupby(self.run_def_cols).apply(t_wrap("input")).reset_index(drop=True)
        df = df.groupby(self.run_def_cols).apply(t_wrap("output")).reset_index(drop=True)
        return df

class TrainMultiplierTransformer(Transformer):

    run_def_cols: List[str] = ['suite_name', 'exp_name', 'run']
    timer_value_types: List[str] = ["", "_bw", "_rounds"]

    timer_id_train: str = "1102"

    batches_per_epoch: Dict[str, int] = {
        "cifar_alexnet": 391,
        "mnist_full": 469,
        "adult": 204,
    }

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        def t(x):
            # check if this group contains an n_batches that is set
            if x['mpc.script_args.n_batches'].isna().any():
                return x

            assert x['mpc.script_args.dataset'].unique().size == 1
            assert x['mpc.script_args.n_batches'].unique().size == 1

            dataset = x['mpc.script_args.dataset'].unique()[0]
            n_batches = int(x['mpc.script_args.n_batches'].unique()[0])
            full_epoch = self.batches_per_epoch[dataset]

            multiplier = full_epoch / n_batches
            print(f"Multiplier: {multiplier} for dataset {dataset} and n_batches {n_batches}")

            for timer_value_type in self.timer_value_types:
                timer_id = f"spdz_timer{timer_value_type}_{self.timer_id_train}"
                if timer_id not in x['stat'].unique():
                    print(f"Skipping run because it does not contain timer {timer_id}")
                    return x
                timer_value = float(x[x['stat'] == timer_id]['stat_value'].values[0])
                timer_value *= multiplier
                x.loc[x['stat'] == timer_id, 'stat_value'] = timer_value

            return x

        df = df.groupby(self.run_def_cols).apply(t).reset_index(drop=True)
        return df

class Sha3MultiplierTransformer(Transformer):

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:

        timer_ids_variable = [
            "90", # TIMER_INPUT_CONSISTENCY_SHA_BIT_DECOMPOSE
            "91", # TIMER_INPUT_CONSISTENCY_SHA_HASH_VARIABLE

            "93",
            "94"
                      ]
        # timer_ids_fixed = [
        #     "92", # TIMER_INPUT_CONSISTENCY_SHA_HASH_FIXED
        # ]
        timer_id_input_consistency = "98" # TIMER_INPUT_CONSISTENCY_CHECK

        def tran(x):
            # access stat_value for stat spdz_timer_98
            for timer_value_type in ["", "_bw", "_rounds"]:
                print("Running for timer_value_type: ", timer_value_type)
                total_time_fixed_old = x[x['stat'] == f"spdz_timer{timer_value_type}_{timer_id_input_consistency}"]['stat_value'].values[0]
                total_time_fixed = total_time_fixed_old
                # print("Total time fixed: ", total_time_fixed)
                total_time_var = 0
                for var in timer_ids_variable:
                    if x[x['stat'] == f"spdz_timer{timer_value_type}_{var}"].empty:
                        continue
                    var_time = float(x[x['stat'] == f"spdz_timer{timer_value_type}_{var}"]['stat_value'].values[0])
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
                x.loc[x['stat'] == f"spdz_timer{timer_value_type}_{timer_id_input_consistency}", 'stat_value'] = total_time_fixed
                print(f"Set from {total_time_fixed_old} to {total_time_fixed} with {multiplier} for timer_value_type: {timer_value_type} and dataset {dataset} {run}")
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


class MetricConfig(MyETLBaseModel):

    class LegendTypeEnum(str, Enum):
        inline = 'inline'
        outside = 'outside'
        hide = 'hide'



    bar_part_cols: List[str]
    y_label: str

    log_y: bool = False

    log_x: bool = False

    y_unit_multiplicator: float = 1.0 # multiply y values by this to get the unit
    y_unit_divider: float = 1.0 # divide y values by this to get the unit

    y_max: float = None

    y_ticks: List[float] = None

    legend_type: LegendTypeEnum = LegendTypeEnum.outside

    legend_order: List[int] = None

    legend_ncol: int = 1

class LegendConfig(MyETLBaseModel):
    format: str
    cols: List[str]

class TitleConfig(MyETLBaseModel):
    format: str
    plot_cols: List[str] # plot cols
class BarPlotLoader(PlotLoader):

    metrics: Dict[str, MetricConfig]

    plot_cols: List[str]
    group_cols: List[str] # for each combination of these clumns, will have a bar group

    # TODO [nku] remove
    group_cols_indices: Optional[List[int]] = None # manual override of the order of the groups, to overlay groups

    bar_cols: List[str] # for each value of these columns, will have a bar in each group

    n_groups_in_bars: int = 1 # number of groups in each bar (the bars in each group can be further divided into groups)
    groups_in_bars_offset: float = 0.02 # the offset between the groups in a bar group

    cols_values_filter: Dict[str, List[str]]

    labels: Dict[str, str]

    legend: LegendConfig

    title: TitleConfig = None

    x_axis_label: str = None



    #metric_cols: List[Union[str, List[str]]]
    #annotation_col: str
    #annotation_labels: Dict[str, str]

    figure_size: List[float] = [2.5, 2.5]

    bar_width: float = 1.2 # 0.6

    show_debug_info: bool = False

    symbols = ['o', 'v']
    colors: List = ['#D5E1A3', '#C7B786', (166 / 255.0, 184 / 255.0, 216 / 255.0), (76 / 255.0, 114 / 255.0, 176 / 255.0), "#5dfc00", "#5dfcf7", "#fd9ef7"]
    color_stack_rgba: List[float] = [1.0, 0.8, 0.6, 0.4, 0.2, 0.2]

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:
        if df.empty:
            return

        # only this should have mb type because it is default spdz output
        df['global_data_sent_mb'] = df['global_data_sent_mb'] * 1000 * 1000
#
        #display(df[["my_overhead", "auditing_overhead_bytes"]])

        output_dir = self.get_output_dir(etl_info)

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

        for metric_name, metric_cfg in self.metrics.items():

            print(f"Creating metric {metric_name} plot...")

            for idx_group, df_plot in df_filtered.groupby(self.plot_cols):

                print(f"  Creating {idx_group} plot...")


                setup_plt(width=self.figure_size[0], height=self.figure_size[1])

                fig, ax = plt.subplots(1, 1)
                # fig.tight_layout()

                if metric_cfg.log_y:
                    ax.set_yscale('log')

                if metric_cfg.log_x:
                    ax.set_xscale('log')

                # allow changing the unit of the metric before calculating the mean / std
                df_plot[metric_cfg.bar_part_cols] = df_plot[metric_cfg.bar_part_cols] * metric_cfg.y_unit_multiplicator / metric_cfg.y_unit_divider

                # check if multiple rows exist for self.group_cols + self.bar_cols
                # if so, output warning
                # create a bar for each bar_cols group in each group_cols group
                grouped_over_reps = df_plot.groupby(by = self.group_cols + self.bar_cols)
                # print first group rows
                # print(f"First group rows: {grouped_over_reps.get_group((list(grouped_over_reps.groups)[0]))}")
                # print(f"First const args: {grouped_over_reps.get_group((list(grouped_over_reps.groups)[0]))['consistency_args.type']}")

                means = grouped_over_reps[metric_cfg.bar_part_cols].mean()
                stds = grouped_over_reps[metric_cfg.bar_part_cols].std()

                # for different bar parts, compute the share (i.e., percent) of the total bar height,
                # and the factor (i.e., how many times larger) the total bar height is compared to the bar part
                means["$total$"] = means[metric_cfg.bar_part_cols].sum(axis=1)
                for col in metric_cfg.bar_part_cols:
                    means[f"$total_share_{col}$"] = means[col] / means["$total$"]
                    means[f"$total_factor_{col}$"] = means["$total$"] / means[col]

                # move bar_columns from index to columns (i.e., pivot)
                unstack_levels = [-i for i in range(len(self.bar_cols), 0, -1)]
                means = means.unstack(unstack_levels)
                stds = stds.unstack(unstack_levels)

                #display(means)
                print(f"MEANS!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(means)


                bar_width = self.bar_width



                def get_label(plot_idx, bar_idx, bar_part):

                    if isinstance(bar_idx, str):
                        bar_idx = [bar_idx]
                    if isinstance(plot_idx, str):
                        plot_idx = [plot_idx]


                    # print(f"Legend Label: plot_idx={plot_idx}   plot_cols={self.plot_cols}  |||||   bar_idx={bar_idx}  bar_cols={self.bar_cols}  |||||  bar_part={bar_part}")


                    subs = []
                    for col in self.legend.cols:

                        if col == "$bar_part_col$":
                            lbl = bar_part

                        elif col in self.bar_cols:
                            idx = self.bar_cols.index(col)
                            lbl = bar_idx[idx]
                        elif col in self.plot_cols:
                            idx = self.plot_cols.index(col)
                            lbl = plot_idx[idx]

                        else:
                            raise ValueError(f"Legend Config Error: col={col} not in self.bar_cols={self.bar_cols} or plot_cols={self.plot_cols}")

                        subs.append(self.labels.get(lbl, lbl)) # label lookup if avbailable

                    return self.legend.format.format(*subs)

                bottom = 0

                # bar_parts_data = []

                for part_idx, bar_part_col in enumerate(metric_cfg.bar_part_cols):

                    means1 = means[bar_part_col]
                    yerr1 = stds[bar_part_col].fillna(0)

                    plot_bar_cols = means1.columns

                    n_groups = len(means1)
                    n_bars_per_group = len(plot_bar_cols)

                    bar_colors = self.colors[:n_bars_per_group]

                    bar_l = np.arange(len(means1))

                    existing_labels = set()


                    for i, col in enumerate(plot_bar_cols):
                        w = bar_width / n_groups # divide by number of rows
                        bar_pos = [j
                                - (w * n_bars_per_group / 2.) # center around j
                                + (i*w) # increment for each column
                                + (w/2.) # center in column
                                for j in bar_l]

                        # within each bar group, we can move the bars into separate groups by introducing an offset
                        if self.n_groups_in_bars == 2:
                            if i < n_bars_per_group / 2:
                                bar_pos = [x - self.groups_in_bars_offset for x in bar_pos]
                            else:
                                bar_pos = [x + self.groups_in_bars_offset for x in bar_pos]
                        elif self.n_groups_in_bars > 2:
                            raise NotImplementedError("n_bars_per_group > 2 not supported")

                        individual_colors_as_rgba = [[mcolors.to_rgba(bar_colors[i], rgba) for _ in range(len(bar_pos))] for rgba in self.color_stack_rgba]


                        label= get_label(plot_idx=idx_group, bar_idx=col, bar_part=bar_part_col)

                        color = individual_colors_as_rgba[part_idx]

                        # only add the label if it does not already exist (together with color)
                        if (color[0], label) in existing_labels:
                            label = None
                        else:
                            existing_labels.add((color[0], label))

                        btm = 0 if part_idx == 0 else bottom[col]

                        ax.bar(bar_pos, means1[col], width=w, label=label, yerr=yerr1[col], color=color, bottom=btm, edgecolor='gray')


                    # bar_parts_data.append(means1)

                    if part_idx == 0:
                        bottom = means1
                    else:
                        bottom += means1

                #total = bottom
                #for x in bar_parts_data:
                #    fraction = total / x
                #    print(f"\n\n\nFraction: {fraction}")


                if  metric_cfg.y_max is not None:
                    ax.set_ylim(0, metric_cfg.y_max)
                if metric_cfg.y_max is None:
                    # increase ylim by 10%
                    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

                if metric_cfg.y_ticks is not None:
                    ax.set_yticks(metric_cfg.y_ticks)


                if  metric_cfg.legend_type != MetricConfig.LegendTypeEnum.hide:
                    # Retrieve current handles and labels for legend
                    handles, labels = plt.gca().get_legend_handles_labels()

                    if metric_cfg.legend_order is not None:
                        # adjust order if specified
                        assert len(handles) == len(metric_cfg.legend_order), f"len(handles)={len(handles)} != len(metric_cfg.legend_order)={len(metric_cfg.legend_order)} for {metric_cfg.bar_part_cols}"
                        handles = [handles[i] for i in metric_cfg.legend_order]
                        labels = [labels[i] for i in metric_cfg.legend_order]

                    if metric_cfg.legend_type == MetricConfig.LegendTypeEnum.outside:  # was (1.05, 1)
                        ax.legend(handles, labels, loc='lower right', bbox_to_anchor=(1.05, 1.1), ncol=metric_cfg.legend_ncol, fancybox=True)
                    elif metric_cfg.legend_type == MetricConfig.LegendTypeEnum.inline:
                        ax.legend(handles, labels, fancybox=True, ncol=metric_cfg.legend_ncol)
                    else:
                        raise NotImplementedError(f"metric_cfg.legend_type={metric_cfg.legend_type}")

                def get_x_tick_labels(means, label_lookup):
                    labels = []
                    for idx in means.index:

                        if isinstance(idx, str):
                            idx = [idx]

                        parts = []
                        for col, value in zip(means.index.names, idx):
                            #print(f"  col={col}  value={value}")
                            if value in label_lookup:
                            #if col in label_lookup and value in label_lookup[col]:
                                #parts.append(label_lookup[col][value])
                                parts.append(label_lookup[value])

                            else:
                                parts.append(value)

                        labels.append("\n".join(parts))

                    return labels

                labels = get_x_tick_labels(means, self.labels)

                pos = range(len(labels))
                ax.set_xticks(pos, labels=labels) #, rotation=90.0


                if self.n_groups_in_bars == 2:
                    n_bars_in_subgroup =  n_bars_per_group / self.n_groups_in_bars
                    w = n_bars_in_subgroup * bar_width / n_groups / 2
                    minor_ticks_pos = [x - w - self.groups_in_bars_offset for x in pos] + [x + w + self.groups_in_bars_offset for x in pos]

                    # TODO [nku] hardcoded
                    minor_ticks_labels = ["Ring" for x in pos] + ["Field" for x in pos]

                    ax.set_xticks(minor_ticks_pos, labels=minor_ticks_labels, minor=True)
                    ax.tick_params(axis='x', which='minor', length=0)

                    ax.tick_params(axis='x', which='major', length=10, width=0, pad=15)



                ax.set_ylabel(metric_cfg.y_label)

                ax.set_xlabel(self.x_axis_label)

                if self.show_debug_info:
                    for p in ax.patches:
                        if hasattr(p, 'get_height') and hasattr(p, 'get_width') and hasattr(p, 'get_x'):
                            ax.annotate(f"{p.get_height():0.2f}", (p.get_x() * 1.005 + (p.get_width() / 2), (p.get_y() + p.get_height()) * 1.005), ha='center', va='bottom')

                        elif hasattr(p, 'get_x') and hasattr(p, 'get_y'):
                            # for lines?
                            ax.annotate(f"{p.get_y():0.2f}", (p.get_x() * 1.005, p.get_y() * 1.005), ha='center', va='bottom')


                if self.title is not None:

                    subs = []
                    for col in self.title.plot_cols:

                        if col in self.plot_cols:
                            idx = self.plot_cols.index(col)
                            lbl = idx_group[idx]
                        else:
                            raise ValueError(f"Title Config Error: col={col} not in self.plot_cols={self.plot_cols}")
                        subs.append(self.labels.get(lbl, lbl)) # label lookup if avbailable

                    title = self.title.format.format(*subs)
                    ax.set_title(title)


                ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

                filename = f"bar_{metric_name}_{escape_tuple_str(idx_group)}"

                out = os.path.join(output_dir, metric_name)
                # make sure out exists
                os.makedirs(out, exist_ok=True)
                self.save_data(means, filename=filename, output_dir=out)
                self.save_plot(fig, filename=filename, output_dir=out, use_tight_layout=True, output_filetypes=["pdf"])