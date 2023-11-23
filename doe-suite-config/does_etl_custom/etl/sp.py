from math import ceil

import pandas as pd
from typing import Dict, List, Union, Optional

from doespy.etl.etl_util import expand_factors, escape_tuple_str
from doespy.etl.steps.transformers import Transformer
from doespy.etl.steps.loaders import Loader, PlotLoader

from does_etl_custom.etl.config import setup_plt

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import colors as mcolors
import matplotlib.container as mcontainer
import numpy as np

COLOR_GRAY = '#999999'

class StatTransformer(Transformer):

    groupby_columns: List[str]

    stats: Dict[str, list]

    def transform(self, df: pd.DataFrame, options: Dict) -> pd.DataFrame:
        groupby_columns = expand_factors(df, self.groupby_columns)

        # Initialize an empty DataFrame to hold the aggregated results
        aggregated_results = pd.DataFrame()

        # check that each stat value specified in self.stats is contained in stat
        for stat_label, stat_values in self.stats.items():
            for stat_value in stat_values:
                if stat_value not in df['stat'].unique():
                    raise ValueError(f"Stat value {stat_value} for stat {stat_label} not found in df['stat'].unique()")

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

            filename = f"consistency_compare_{escape_tuple_str(idx)}"
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False)


class BarPlotLoader(PlotLoader):

    plot_cols: List[str]
    plot_cols_values: Dict[str, List[str]]

    group_cols: List[str] = ["workload_composition_mode"] # for each combination of these clumns, will have a bar group
    group_cols_values: Dict[str, List[str]] = {"workload_composition_mode": ["upc-block-composition", "poisson-block-composition-pa", "poisson-block-composition"]}
    group_cols_indices: Optional[List[int]] = None # manual override of the order of the groups, to overlay groups

    bar_cols: List[str] = ["allocation"] # for each value of these columns, will have a bar in each group
    bar_cols_values: Dict[str, List[str]] = {"allocation": ['greedy', "weighted-dpf+", "dpk-gurobi"]}


    metric_cols: List[str]
    annotation_col: str
    annotation_labels: Dict[str, str]

    show_debug_info: bool = True

    symbols = ['o', 'v']
    colors: List = ['#D5E1A3', '#C7B786', (166 / 255.0, 184 / 255.0, 216 / 255.0), (76 / 255.0, 114 / 255.0, 176 / 255.0), "#5dfc00", "#5dfcf7", "#fd9ef7"]

    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:
        if df.empty:
            return

        output_dir = self.get_output_dir(etl_info)

        n_rows_intial = len(df)
        df_filtered = df.copy()

        plot_cols = [(col, self.plot_cols_values[col]) for col in self.plot_cols]
        group_cols = [(col, self.group_cols_values[col]) for col in self.group_cols]
        bar_cols = [(col, self.bar_cols_values[col]) for col in self.bar_cols]
        # filter out non-relevant results
        for col, allowed in plot_cols + group_cols + bar_cols:

            # convert column to string for filtering
            df_filtered[col] = df_filtered[col].astype(str)

            print(f"Filtering {col} to {allowed}    all={df[col].unique()}")
            # filter out non-relevant results
            df_filtered = df_filtered[df_filtered[col].isin(allowed)]
            # convert to categorical
            df_filtered[col] = pd.Categorical(df_filtered[col], ordered=True, categories=allowed)
        df_filtered.sort_values(by=self.plot_cols + self.group_cols + self.bar_cols, inplace=True)

        print(f"Filtered out {n_rows_intial - len(df_filtered)} rows (based on plot_cols, row_cols, col_cols)  remaining: {len(df_filtered)}")

        for metric in self.metric_cols:
            for idx, df_plot in df_filtered.groupby(self.plot_cols):
                print(f"Creating Workload {idx} plot")

                setup_plt()
                fig, ax = plt.subplots(1, 1)
                # fig.tight_layout()

                # create a bar for each bar_cols group in each group_cols group
                grouped_over_reps = df_plot.groupby(by = self.group_cols + self.bar_cols)
                means = grouped_over_reps[metric].mean()
                stds = grouped_over_reps[metric].std()

                # move bar_columns from index to columns (i.e., pivot)
                unstack_levels = [-i for i in range(len(self.bar_cols), 0, -1)]
                means = means.unstack(unstack_levels)
                stds = stds.unstack(unstack_levels)


                ###################################
                # Drawing the utility bar/scatter #
                ###################################
                # Create bar chart
                bar_width = 0.25

                # setup index map
                index_map = {}
                idx = 0


                # TODO [nku] could do nicer
                n_groups = len(means)
                n_bars_per_group = len(means.columns)
                bar_colors = self.colors[:n_bars_per_group]
                color_positions = n_groups * bar_colors


                for index, _row in means.iterrows():

                    if isinstance(index, str):
                        index = [index]


                    for column in means.columns:
                        if isinstance(column, str):
                            column = [column]

                        k = tuple(tuple(index) + tuple(column))

                        index_map[k] = idx
                        idx+=1

                # TODO [nku] could bring back legend=False if we have custom legend handling below
                # legend=False
                yerr = stds.fillna(0)
                # container = means.plot.bar(yerr=yerr, ax=ax, width=bar_width, color=bar_colors)

                # Use matplotlib to plot bars, a group for each index and a bar in each group for each column
                # Get the number of columns
                num_of_cols = len(means.columns)

                # Create an array with the positions of each bar on the x-axis
                bar_l = np.arange(len(means))

                # Make the bar chart
                for i, col in enumerate(means.columns):
                    w = bar_width / len(means) # divide by number of columns
                    bar_pos = [j
                               - (w * num_of_cols / 2.) # center around j
                               + (i*w) # increment for each column
                               + (w/2.) # center in column
                               for j in bar_l]

                    individual_colors_as_rgba = [mcolors.to_rgba(bar_colors[i], 1.0) for _ in range(len(bar_pos))]

                    # If adjacent bar_pos are the same, low alpha of the first bar to make it transparent
                    # for j in range(len(bar_pos)-1):
                    #     if bar_pos[j] == bar_pos[j+1]:
                    #         individual_colors_as_rgba[j] = lighten_color(individual_colors_as_rgba[j], amount=1.4)

                    ax.bar(bar_pos, means[col], width=w, label=col, yerr=yerr[col], color=individual_colors_as_rgba)

                    # extract x positions of the bars + add  bar_width/8. to get the position for the circle
                    x_positions = [None] * (len(means.columns) * len(means))
                    container_id = 0
                    for c in ax.containers:
                        # I think containers are the individual calls to ax.bar ??
                        if isinstance(c, mcontainer.BarContainer):
                            for bar_id, rect in enumerate(c.patches):
                                # fill x_positions in horizontal absolute order of the bars
                                x_positions[bar_id * len(means.columns) + container_id] = rect.get_x() + bar_width/(n_bars_per_group * 2.)
                            container_id += 1

                    # determine if this x_positions is in a group, and if yes, if it is first or second
                    # 0 = no group, 1 = first in group, 2 = second in group
                    x_position_in_group = [0] * len(x_positions)
                    for i in range(len(bar_pos)-1):
                        if bar_pos[i] == bar_pos[i+1]:
                            for j in range(len(means)):
                                x_position_in_group[(i * len(means.columns)) + j] = 1
                                x_position_in_group[((i+1) * len(means.columns)) + j] = 2


                ax.set_xticks(bar_l)
                ax.set_xticklabels(means.index)
                def get_x_tick_labels(means, label_lookup):
                    labels = []
                    for idx in means.index:

                        if isinstance(idx, str):
                            idx = [idx]

                        parts = []
                        for col, value in zip(means.index.names, idx):
                            #print(f"  col={col}  value={value}")
                            if col in label_lookup and value in label_lookup[col]:
                                parts.append(label_lookup[col][value])
                            else:
                                parts.append(value)

                        labels.append("\n".join(parts))

                    return labels

                labels = get_x_tick_labels(means, {})

                pos = range(len(labels))
                ax.set_xticks(pos, labels=labels) #, rotation=90.0

                # Rotate the tick labels to be horizontal
                ax.tick_params(axis='x', labelrotation=0)

                # Reduce space on both sides of x-axis to allow for more bar space
                ax.set_xlim(min(x_positions)-0.25, max(x_positions)+0.25)

                # increase ylim by 30%
                ax.set_ylim(0, ax.get_ylim()[1] * 1.5)

                ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

                ax.set_ylabel(metric)

                if self.show_debug_info:
                    for p in ax.patches:
                        if hasattr(p, 'get_height') and hasattr(p, 'get_width') and hasattr(p, 'get_x'):
                            ax.annotate(f"{p.get_height():0.2f}", (p.get_x() * 1.005 + (p.get_width() / 2), p.get_height() * 1.005), ha='center', va='bottom')

                        elif hasattr(p, 'get_x') and hasattr(p, 'get_y'):
                            # for lines?
                            ax.annotate(f"{p.get_y():0.2f}", (p.get_x() * 1.005, p.get_y() * 1.005), ha='center', va='bottom')

                handles, legend_labels = ax.get_legend_handles_labels()

                # Legend
                if True:
                    if len(self.bar_cols) == 1:
                        single_key = bar_cols[0]
                        labels = legend_labels
                        # if label_lookup is not None and single_key in label_lookup:
                        #     labels = [label_lookup[single_key][legend_label] for legend_label in legend_labels]
                        # else:
                        #     labels = legend_labels
                        # ax.legend(labels=labels, handles=handles, bbox_to_anchor=legend_bbox_to_anchor_map.get(workload_name), loc=3)
                        ax.legend(labels=labels, handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0))

                # plt.tight_layout()
                plt.subplots_adjust(left=0.2)

                filename = f"metrics_compare_{metric}_{escape_tuple_str(idx)}"
                self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=False)