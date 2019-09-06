# chart
import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter

from db import Db
from my_utils import Misc

logger = logging.getLogger('thermo.' + 'chart')


class Chart:

    @classmethod
    def meter_chart(cls, meters, dtime, temper, atmo, delta_atmo, chart_fname_path,
                    ranges=None, alert=None, show=False):
        meter_id = meters[0]  # chart for one meter only is implemented now
        fig, ax1 = plt.subplots()
        # fig.suptitle(f'Динаміка температура для {name}')
        if alert is not None:
            ax1.set_title(alert)
        else:
            ax1.set_title(f'Динаміка показників температури для місця {meter_id}')

        myFmt = DateFormatter("%m-%d %H:%M:%S")

        linestyle_temp = 'dotted'
        linestyle_diff = 'solid'

        # temperature:
        color_temp = 'tab:red'
        temp_plot, = ax1.plot(dtime, temper, color=color_temp, linestyle=linestyle_temp, marker='.',
                              label=f'Температура')
        ax1.set_xlabel('Час')
        ax1.xaxis.set_major_formatter(myFmt)
        # ax1.xaxis.set_major_locator(ticker.LinearLocator(35))
        ax1.grid(axis='x')
        # fig.autofmt_xdate()
        ax1.xaxis.set_tick_params(labelsize=8, rotation=45)

        # atmo
        color = 'tab:green'
        ax1.set_ylabel(f'Температура(\'C)', color=color)
        atmo_plot, = ax1.plot(dtime, atmo, color=color, linestyle=linestyle_temp, label=f'Темп.оточення')
        # ax1.legend(loc='best') # upper left')

        # atmo diff
        color = 'tab:blue'
        ax2 = ax1.twinx()
        ax2.set_ylabel(f'Різниця(\'C) з оточенням', color=color)
        ax2.xaxis.set_major_formatter(myFmt)
        ax2.xaxis.set_major_locator(ticker.LinearLocator(15))
        diff_plot, = ax2.plot(dtime, delta_atmo, color=color, linestyle=linestyle_diff, marker='.',
                              label=f'Різниця з оточ.')
        # ax2.legend(loc='best') # 'upper right')
        ax2.legend(handles=(temp_plot, atmo_plot, diff_plot))

        # set range lines:
        if ranges is not None:
            temp_yellow, temp_red, atmo_yellow, atmo_red, _, _ = ranges
            ax1.axhline(temp_yellow, color='tab:orange', linestyle=linestyle_temp, linewidth=0.5)
            ax1.axhline(temp_red, color='tab:red', linestyle=linestyle_temp, linewidth=0.5)
            ax2.axhline(atmo_yellow, color='tab:orange', linestyle=linestyle_diff, linewidth=0.5)
            ax2.axhline(atmo_red, color='tab:red', linestyle=linestyle_diff, linewidth=0.5)

        fig.tight_layout()
        fig.savefig(chart_fname_path)
        if show:
            plt.show()


if __name__ == '__main__':

    data_sets = {
        # name : (name, (meter_id,...), start_dtime, end_dtime)
        'd1': ('d1', (1041,), '20190824T155400', '20190824T165600'),
        'd2': ('d2', (1041,), '20190824T181000', '20190824T181900'),
        'd3': ('d3', (1004,), '20190824T183700', '20190824T185900'),
        'd4': ('d4', (1004,), '20190826T114741', '20190826T122738')
    }


    def load_readings(start_dtime=None, end_dtime=None):
        if not start_dtime and not end_dtime:
            dtime_condition, query_args = '', ()
        elif start_dtime and not end_dtime:
            dtime_condition, query_args = 'dtime >= ?', (start_dtime,)
        elif not start_dtime and end_dtime:
            dtime_condition, query_args = 'dtime <= ?', (end_dtime,)
        elif start_dtime and end_dtime:
            dtime_condition, query_args = '(dtime >= ? and dtime <=?)', (start_dtime, end_dtime)
        else:
            print(f'are you kidding me? start={start_dtime} end={end_dtime}')
            exit(1)
        where_clause = f'where {dtime_condition} '  # {"and " if dtime_condition else " "}  avg_atmo is not null '

        query = f'select * from Readings_hist ' \
                f'{where_clause} ' \
                f'order by dtime'
        logger.debug(f'query={query}, args={query_args}')
        readings = Db.select(query, query_args, Db.HistReadingsRecord)
        logger.debug(f'loaded {len(readings)} readings, '
                     f'dtime range: {min([r.dtime for r in readings])} : {max([r.dtime for r in readings])}')
        return readings


    def chart_meter(name, meters, start_dtime, end_dtime):
        pass


    def main():
        ds_name = 'd4'
        logger.debug(f'data set name={ds_name}')
        Chart.meter_chart(*data_sets[ds_name])


    main()
