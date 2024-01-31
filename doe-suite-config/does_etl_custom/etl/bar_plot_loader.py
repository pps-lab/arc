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

class MetricConfig(MyETLBaseModel):

    bar_part_cols: List[str]
    y_label: str

    # TODO [nku]: CAN I MOVE THESE PARTS AWAY FROM THE METRIC CONFIG AND INTO A FILTER SYSTEM SIMILAR TO THE BAR_PLOT_STYLE?

    log_y: bool = False

    log_x: bool = False


    y_unit_multiplicator: float = 1.0 # multiply y values by this to get the unit
    y_unit_divider: float = 1.0 # divide y values by this to get the unit

    y_max: float = None

    y_ticks: List[float] = None


    x_lim: Tuple[float, float] = None
    y_lim: Tuple[float, float] = None

    # TODO: remove hack
    y_lim_row: List[Tuple[float, float]] = None


    plot_cols_filter: Dict[str, List[str]] = None

    bar_cols_filter: Dict[str, List[str]] = None

    bar_pos_bias: float = 0.0




class LegendConfig(MyETLBaseModel):
    format: str
    cols: List[str]

    kwargs: Dict[str, Any] = None


class AxLegendConfig(LegendConfig):

    filter: Dict[str, List[str]] = None


def get_label_cols_and_format(full_id, legend_configs: List[AxLegendConfig]):

    for cfg in legend_configs:
        # check filter
        is_match = True
        for col, values in cfg.filter.items():
            if col not in full_id:
                raise ValueError(f"Error in legend_ax, the filter column: {col} is not avaialable ({full_id.keys()})")
            elif full_id[col] not in values:
                is_match = False
                break

        # apply style
        if is_match:
            return cfg.cols, cfg.format

    raise ValueError(f"Error in legend_ax, no match found for {full_id}")


def style_ax_legend(ax, plot_id, legend_configs: List[AxLegendConfig]):


    for cfg in legend_configs:

        if cfg.kwargs is None: # only consider ones with kwargs
            continue

        # check filter
        is_match = True
        for col, values in cfg.filter.items():
            if col not in plot_id:
                raise ValueError(f"Error in legend_ax, the filter column: {col} is not avaialable ({full_id.keys()})")
            elif plot_id[col] not in values:
                is_match = False
                break

        if is_match:
            # collect lables and handles
            handles, labels = [], []
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.append(handle)
                labels.append(label)

            # remove duplicates
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]

            ax.legend(*zip(*unique), **cfg.kwargs)

            return

    raise ValueError(f"Error in legend_ax, no match found for {plot_id}")



class FigLegendConfig(LegendConfig):

    subplot_idx: Tuple[int, int] = None


    def style_fig_legend(self, fig, axs):

        # collect lables and handles
        handles, labels = [], []
        for ax in axs.flat:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                handles.append(handle)
                labels.append(label)

        # remove duplicates
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]

        container = fig if self.subplot_idx is None else axs[self.subplot_idx[0], self.subplot_idx[1]]
        container.legend(*zip(*unique), **self.kwargs)



class TitleConfig(MyETLBaseModel):
    format: str
    plot_cols: List[str] # plot cols


class TickLabelConfig(MyETLBaseModel):
    format: str
    cols: List[str]

class SubplotConfig(MyETLBaseModel):
    rows: List[str]
    cols: List[str]

    share_x: Literal['none', 'all', 'row', 'col'] = 'none'
    share_y: Literal['none', 'all', 'row', 'col'] = 'row'



class BarStyle(MyETLBaseModel):

    style: Dict[str, typing.Any]
    filter: Dict[str, List[str]]


def get_bar_style(bar_styles: List[BarStyle], full_id):

    if bar_styles is None:
        return {}

    config = {}

    for style in bar_styles:

        # check filter
        is_match = True
        for col, values in style.filter.items():
            if col not in full_id:
                raise ValueError(f"Error in BarStyle, the filter column: {col} is not avaialable ({full_id.keys()})")
            elif full_id[col] not in values:
                is_match = False
                break

        # apply style
        if is_match:
            for k, v in style.style.items():
                if k not in config:
                    config[k] = v

    return config




class BarPlotLoader(PlotLoader):

    metrics: Dict[str, MetricConfig]

    subplots: SubplotConfig = None

    plot_cols: List[str]
    group_cols: List[str] # for each combination of these clumns, will have a bar group

    bar_cols: List[str] # for each value of these columns, will have a bar in each group

#    n_groups_in_bars: int = 1 # number of groups in each bar (the bars in each group can be further divided into groups)
#    groups_in_bars_offset: float = 0.02 # the offset between the groups in a bar group

    cols_values_filter: Dict[str, List[str]]

    labels: Dict[str, str]

    legend_fig: FigLegendConfig = None
    legend_ax: List[AxLegendConfig] = None

    title: TitleConfig = None

    x_axis_label: str = None

    group_labels: TickLabelConfig = None
    # bar_labels: TickLabelConfig = None


    bar_styles: List[BarStyle] = None

    figure_size: List[float] = [2.5, 2.5]

    bar_width: float = 1.2 # 0.6

    show_debug_info: bool = False


    def load(self, df: pd.DataFrame, options: Dict, etl_info: Dict) -> None:
        if df.empty:
            return

        output_dir = self.get_output_dir(etl_info)

        df_filtered = self.filter_df(df)

        if self.subplots is not None:
            subplots = Subplots()
            fig, axs = subplots.init_subplots(df=df_filtered, loader=self)

        for metric_name, metric_cfg in self.metrics.items():

            print(f"Creating metric {metric_name} plot...")

            for idx_plot, df_plot in df_filtered.groupby(self.plot_cols):

                idx_plot = (idx_plot, ) if not isinstance(idx_plot, tuple) else idx_plot
                plot_id = {k: v for k, v in zip(self.plot_cols, list(idx_plot), strict=True)}

                print(f"  Creating {plot_id} plot...")

                if self.subplots is None:
                    setup_plt(width=self.figure_size[0], height=self.figure_size[1])

                    fig, ax = plt.subplots(1, 1)
                    subplot_idx = (0, 0)
                else:

                    subplot_idx = subplots.get_subplot_idx(plot_id, metric_name)

                    if subplot_idx is None:
                        continue

                    ax = axs[subplot_idx[0], subplot_idx[1]]


                df1 = self.aggregate_data(df_plot, metric_cfg)

                group_ticks, bar_ticks = self.plot_bars(ax, metric_cfg, plot_id, df1)

                if self.legend_ax is not None:
                    style_ax_legend(ax, plot_id, self.legend_ax)

                self.style_y_axis(ax, metric_cfg, subplot_idx)

                self.style_x_axis(ax, metric_cfg, subplot_idx, group_ticks, bar_ticks)

                self.debug_info(ax)

                self.style_title(ax, metric_cfg, plot_id, subplot_idx)

                ax.grid(True, axis="y", linestyle=':', color='0.6', zorder=0, linewidth=1.2)

                filename = f"bar_{metric_name}_{escape_tuple_str(idx_plot)}"

                out = os.path.join(output_dir, metric_name)
                os.makedirs(out, exist_ok=True)

                self.save_data(df1, filename=filename, output_dir=out)

                if self.subplots is None:
                    self.save_plot(fig, filename=filename, output_dir=out, use_tight_layout=True, output_filetypes=["pdf"])


        if self.legend_fig is not None and self.subplots is not None:
            self.legend_fig.style_fig_legend(fig, axs)


        if self.subplots is not None:
            filename =  f"{etl_info['pipeline']}_debug" if self.show_debug_info else etl_info['pipeline']
            self.save_plot(fig, filename=filename, output_dir=output_dir, use_tight_layout=True, output_filetypes=["pdf"])


    def debug_info(self, ax):

        if self.show_debug_info:
            for p in ax.patches:
                if hasattr(p, 'get_height') and hasattr(p, 'get_width') and hasattr(p, 'get_x'):
                    ax.annotate(f"{p.get_height():0.2f}", (p.get_x() * 1.005 + (p.get_width() / 2), (p.get_y() + p.get_height()) * 1.005), ha='center', va='bottom')

                elif hasattr(p, 'get_x') and hasattr(p, 'get_y'):
                    # for lines?
                    ax.annotate(f"{p.get_y():0.2f}", (p.get_x() * 1.005, p.get_y() * 1.005), ha='center', va='bottom')



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



    def style_y_axis(self, ax, metric_cfg, subplot_idx: Tuple[int, int]):

        #ax.tick_params(axis='y', labelsize=14)  # You can adjust the size according to your preference


        if metric_cfg.log_y:
            ax.set_yscale('log')




        if  metric_cfg.y_max is not None:
            ax.set_ylim(0, metric_cfg.y_max)
        else:
            if metric_cfg.log_y:
                pass
                #ax.set_ylim(0, ax.get_ylim()[1] * 2)
            else:
                pass # problematic for subplots with shared axis
                #ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

        if metric_cfg.y_lim is not None:
            ax.set_ylim(*metric_cfg.y_lim)

        if metric_cfg.y_lim_row is not None:

            assert self.subplots is not None, "y_lim_row only supported for subplots"

            col = subplot_idx[1]

            ax.set_ylim(*metric_cfg.y_lim_row[col])



        if metric_cfg.y_ticks is not None:
            ax.set_yticks(metric_cfg.y_ticks)



        if self.subplots is None:
            ax.set_ylabel(metric_cfg.y_label)
        elif subplot_idx[1] == 0:
            ax.set_ylabel(metric_cfg.y_label)

        # TODO [nku] make configurable -> does not work?
        #ax.tick_params(axis='y', which='minor', bottom=False)
        #ax.tick_params(axis='y', which='major', bottom=False)



    def style_x_axis(self, ax, metric_cfg, subplot_idx: Tuple[int, int], group_ticks, bar_ticks):
        if metric_cfg.log_x:
            ax.set_xscale('log')


        # by default use all group_columns for the ticks
        cols, format = (self.group_cols, " ".join([r"{}"] * len(self.group_cols))) if self.group_labels is None else (self.group_labels.cols, self.group_labels.format)

        positions = []
        labels = []

        for pos, d in group_ticks:

            subs = []
            for col in cols:
                if col not in d:
                    raise ValueError(f"Error: Group Label Config:  col={col} not available: {d}")
                subs.append(self.labels.get(d[col], d[col])) # label lookup if avbailable


            tick_label = format.format(*subs)

            positions.append(pos)
            labels.append(tick_label)

            ax.set_xticks(positions, labels=labels)




        #if self.bar_labels is not None: # bar tick labels (minor)
        #    pass

            # NOTE: IN COMBINATION WITH SUBGROUPS THE BAR LABELS ALSO NEED TO BE SHIFTED
            #if self.n_groups_in_bars == 2:
            #        n_bars_in_subgroup =  n_bars_per_group / self.n_groups_in_bars
            #        w = n_bars_in_subgroup * self.bar_width / n_bar_groups / 2
            #        minor_ticks_pos = [x - w - self.groups_in_bars_offset for x in pos] + [x + w + self.groups_in_bars_offset for x in pos]
#
            #        # TODO [nku] hardcoded
            #        minor_ticks_labels = ["Ring" for x in pos] + ["Field" for x in pos]
#
            #        ax.set_xticks(minor_ticks_pos, labels=minor_ticks_labels, minor=True)
            #        ax.tick_params(axis='x', which='minor', length=0)
#
            #        ax.tick_params(axis='x', which='major', length=10, width=0, pad=15)

        ax.set_xlabel(self.x_axis_label)

        if metric_cfg.x_lim is not None:
            ax.set_xlim(*metric_cfg.x_lim)

        #print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!x_lim: {ax.get_xlim()}")


#    def style_fig_legend(self, fig, axs):
#
#        if self.subplots is not None and self.subplots.share_legend:
#
#            handles, labels = [], []
#
#            for ax in axs.flat:
#                for handle, label in zip(*ax.get_legend_handles_labels()):
#                    handles.append(handle)
#                    labels.append(label)
#
#            # Remove duplicates
#            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
#
#            # Create legend
#            #  bbox_to_anchor=(1.05, 1.1)
#
#            #fig.legend(*zip(*unique), loc='upper center', ncol=len(unique), bbox_to_anchor=(0.51, 0.075), columnspacing=3.5, fancybox=True) #Skipping plot because
#
#            axs[0, 0].legend(*zip(*unique), loc='best', ncol=1, fancybox=True)
#
#    def style_ax_legend(self, ax, metric_cfg):
#
#
#        if self.subplots is not None and self.subplots.share_legend:
#            # Build a single legend outside the plot
#            return
#
#
#        if  metric_cfg.legend_type != MetricConfig.LegendTypeEnum.hide:
#            # Retrieve current handles and labels for legend
#
#            # TODO [nku] MIGHT NEED TO DO ax.get_legend_handles_labels() INSTEAD -> LEGEND MAY ALSO NEED TO BE DONE PER PLOT OR PER SUBPLOT
#
#            handles, labels = ax.get_legend_handles_labels()
#
#            if metric_cfg.legend_order is not None:
#                # adjust order if specified
#                assert len(handles) == len(metric_cfg.legend_order), f"len(handles)={len(handles)} != len(metric_cfg.legend_order)={len(metric_cfg.legend_order)} for {metric_cfg.bar_part_cols}"
#                handles = [handles[i] for i in metric_cfg.legend_order]
#                labels = [labels[i] for i in metric_cfg.legend_order]
#
#            if metric_cfg.legend_type == MetricConfig.LegendTypeEnum.outside:  # was (1.05, 1)
#                ax.legend(handles, labels, loc='lower right', bbox_to_anchor=(1.05, 1.1), ncol=metric_cfg.legend_ncol, fancybox=True)
#            elif metric_cfg.legend_type == MetricConfig.LegendTypeEnum.inline:
#                ax.legend(handles, labels, fancybox=True, ncol=metric_cfg.legend_ncol)
#            else:
#                raise NotImplementedError(f"metric_cfg.legend_type={metric_cfg.legend_type}")
#
    def style_title(self, ax, metric_cfg, plot_id, subplot_idx):

        if self.title is not None:

            subs = []
            for col in self.title.plot_cols:
                if col not in plot_id:
                    raise ValueError(f"Title Config Error: col={col} not in self.plot_cols={self.plot_cols}")

                lbl = plot_id[col]
                subs.append(self.labels.get(lbl, lbl)) # label lookup if avbailable

            title = self.title.format.format(*subs)

            if self.subplots is None:
                ax.set_title(title)
            elif subplot_idx[0] == 0:
                ax.set_title(title)



    def plot_bars(self, ax, metric_cfg, plot_id, df1):
        existing_labels = set()

        n_bars_per_group, n_bar_groups = self.get_bars_info(df1)


        w = self.bar_width / n_bar_groups # divide by number of rows

        group_ticks = []
        bar_ticks = []


        group_cols = self.group_cols[0] if len(self.group_cols) == 1 else self.group_cols
        for i_group, (idx_bar_group, df_bar_group) in enumerate(df1.groupby(group_cols)):
            idx_bar_group = (idx_bar_group, ) if not isinstance(idx_bar_group, tuple) else idx_bar_group


            bar_group_id = {k: v for k, v in zip(self.group_cols, list(idx_bar_group), strict=True)}


            group_pos_center = np.arange(n_bar_groups)[i_group]

            group_pos_left = group_pos_center - (w * n_bars_per_group / 2.)

            group_ticks.append((group_pos_center, {**plot_id, **bar_group_id}))

            bar_cols = self.bar_cols[0] if len(self.bar_cols) == 1 else self.bar_cols
            for i_bar, (idx_bar, df_bar) in enumerate(df_bar_group.groupby(bar_cols)):
                idx_bar = (idx_bar, ) if not isinstance(idx_bar, tuple) else idx_bar

                bar_id = {k: v for k, v in zip(self.bar_cols, list(idx_bar), strict=True)}


                # TODO [nku] should be unified and also should be available for groups
                # skipping plots that are not in the filter
                if metric_cfg.bar_cols_filter is not None:
                    skip = False
                    for col, allowed in metric_cfg.bar_cols_filter.items():
                        if bar_id[col] not in allowed:
                            skip = True
                    if skip:
                        print(f"    Skipping bar because {bar_id} does not match metric filter: {metric_cfg.bar_cols_filter}")
                        continue

                # increment for each column + center
                bar_pos = group_pos_left + (i_bar*w) + (w/2.)


                # TODO: could be made nicer -> but would require a lot more complex bar positioning logic
                bar_pos += metric_cfg.bar_pos_bias * w

                # TODO: this bar position is probably wrong

                bar_ticks.append((bar_pos, {**plot_id, **bar_group_id, **bar_id}))


#           NOTE: At some point, there was code to support subgroups within bar groups
#                        # within each bar group, we can move the bars into separate groups by introducing an offset
#                        if self.n_groups_in_bars == 2:
#                            if i < n_bars_per_group / 2:
#                                bar_pos = [x - self.groups_in_bars_offset for x in bar_pos]
#                            else:
#                                bar_pos = [x + self.groups_in_bars_offset for x in bar_pos]
#                        elif self.n_groups_in_bars > 2:
#                            raise NotImplementedError("n_bars_per_group > 2 not supported")
#


                bottom = 0

                for bar_part_col in metric_cfg.bar_part_cols:
                    bar_part_id = {"$bar_part_col$": bar_part_col}

                    yerr = df_bar[bar_part_col]["std"].fillna(0) # NOTE: should we do this?


                    full_id = {**plot_id, **bar_group_id, **bar_id, **bar_part_id}


                    style_config = get_bar_style(self.bar_styles, full_id)


                    label = style_config.pop("label", self.get_label(full_id))


                    #color=color, edgecolor='black', linewidth=1

                    ax.bar(bar_pos, df_bar[bar_part_col]["mean"], width=w, label=label, yerr=yerr,  bottom=bottom, **style_config)

                    bottom += df_bar[bar_part_col]["mean"]

        return group_ticks, bar_ticks


    def get_label(self, full_id):


        if self.legend_fig is None and self.legend_ax is None:
            return None

        assert self.legend_fig is None or self.legend_ax is None, "Cannot have both fig and ax legend"

        cols, format = (self.legend_fig.cols, self.legend_fig.format) if self.legend_ax is None else get_label_cols_and_format(full_id, self.legend_ax)

        subs = []
        for col in cols:

            if col not in full_id:
                raise ValueError(f"Legend Config Error: col={col} not available: {full_id}")

            lbl = full_id[col]
            subs.append(self.labels.get(lbl, lbl)) # label lookup if available

        return format.format(*subs)


    def get_bars_info(self, combined_new):
        n_bars_info = []

        group_cols = self.group_cols[0] if len(self.group_cols) == 1 else self.group_cols
        for _, df_bar_group in combined_new.groupby(group_cols):
            bar_cols = self.bar_cols[0] if len(self.bar_cols) == 1 else self.bar_cols
            n_bars_per_group = df_bar_group.groupby(bar_cols).ngroups
            n_bars_info.append(n_bars_per_group)

        n_bar_groups = len(n_bars_info)
        assert all(x == n_bars_info[0] for x in n_bars_info), "not all bar groups have the same number of bars"
        n_bars_per_group = n_bars_info[0]

        return n_bars_per_group,n_bar_groups




class Subplots:

    def init_subplots(self, df, loader: BarPlotLoader):

        self.loader = loader

        row_keys = set()
        col_keys = set()

        for metric_name, metric_cfg in loader.metrics.items():
            for idx_plot, _ in df.groupby(loader.plot_cols):

                plot_id = {k: v for k, v in zip(loader.plot_cols, list(idx_plot), strict=True)}

                # skipping plots that are not in the filter
                if metric_cfg.plot_cols_filter is not None:
                    skip = False
                    for col, allowed in metric_cfg.plot_cols_filter.items():
                        if plot_id[col] not in allowed:
                            skip = True
                    if skip:
                        print(f"    Skipping plot because {idx_plot} does not match metric filter: {metric_cfg.plot_cols_filter}")
                        continue

                row_keys.add(self._lookup_key(plot_id, metric_name, relevant_columns=loader.subplots.rows))
                col_keys.add(self._lookup_key(plot_id, metric_name, relevant_columns=loader.subplots.cols))

        def _build_lookup(allowed_keys, relevant_columns):
            lookup = {}
            i = 0
            cross = [loader.metrics.keys() if x == "$metrics$" else loader.cols_values_filter[x] for x in relevant_columns]
            for k in itertools.product(*cross):
                if k in allowed_keys:
                    lookup[k] = i
                    i += 1
            return lookup

        self._row_lookup = _build_lookup(allowed_keys=row_keys, relevant_columns=loader.subplots.rows)
        self._col_lookup = _build_lookup(allowed_keys=col_keys, relevant_columns=loader.subplots.cols)

        nrows = len(self._row_lookup)
        ncols = len(self._col_lookup)
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * loader.figure_size[0], nrows * loader.figure_size[1]), sharex=loader.subplots.share_x, sharey=loader.subplots.share_y)
        return fig, axs


    def get_subplot_idx(self, plot_id, metric_name):
        row_key = self._lookup_key(plot_id, metric_name, relevant_columns=self.loader.subplots.rows)
        col_key = self._lookup_key(plot_id, metric_name, relevant_columns=self.loader.subplots.cols)


        if row_key not in self._row_lookup or col_key not in self._col_lookup:
            print(f"Skipping plot because {row_key} or {col_key} not in {self._row_lookup} or {self._col_lookup}")
            return None
        else:
            return (self._row_lookup[row_key], self._col_lookup[col_key])


    def _lookup_key(self, plot_id, metric_name, relevant_columns):
        key = []
        for x in relevant_columns:
            if x == "$metrics$":
                key.append(metric_name)
            else:
                key.append(plot_id[x])
        return tuple(key)
