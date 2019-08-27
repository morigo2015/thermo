# chart
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter

from db import Db
from my_utils import Misc

logger = logging.getLogger('thermo.' + 'chart')

data_sets = {
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
    readings = Db.select(query, query_args, Db.ReadingsHistRecord)
    logger.debug(f'loaded {len(readings)} readings, '
                 f'dtime range: {min([r.dtime for r in readings])} : {max([r.dtime for r in readings])}')
    return readings


def chart_meter(name, meters, start_dtime, end_dtime):
    meter_id = meters[0]
    readings = load_readings(start_dtime, end_dtime)
    selected_readings = [r for r in readings if r.meter_id == meter_id]  # or meter_id == -1]
    if not selected_readings:
        print(f'empty list of readings!! name={name} meter_id={meter_id}, start={start_dtime} end={end_dtime}')
        return
    dtime, temper, atmo, delta_atmo = list(zip(*[
        (Misc.str_to_dtime(r.dtime), r.temperature, r.avg_atmo, (r.temperature - r.avg_atmo) if r.avg_atmo else None)
        for r in selected_readings]))

    fig, ax1 = plt.subplots()
    # fig.suptitle(f'Динаміка температура для {name}')
    ax1.set_title(f'Динаміка температури для сценарію "{name}"')
    myFmt = DateFormatter("%m-%d %H:%M:%S")

    color = 'tab:red'
    temp_plot, =ax1.plot(dtime, temper, color=color, label=f'Температура {meter_id}')
    ax1.set_xlabel('Час')
    ax1.xaxis.set_major_formatter(myFmt)
    # ax1.xaxis.set_major_locator(ticker.LinearLocator(35))
    ax1.grid(True)
    # fig.autofmt_xdate()
    ax1.xaxis.set_tick_params(labelsize=8, rotation=45)

    color = 'tab:green'
    ax1.set_ylabel(f'Температура (власна)', color=color)
    atmo_plot, =ax1.plot(dtime, atmo, color=color, label=f'Темп.оточення {meter_id}')
    # ax1.legend(loc='best') # upper left')

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'Різниця з оточенням', color=color)
    ax2.xaxis.set_major_formatter(myFmt)
    ax2.xaxis.set_major_locator(ticker.LinearLocator(15))
    diff_plot, =ax2.plot(dtime, delta_atmo, color=color, label=f'Різниця {meter_id}')
    # ax2.legend(loc='best') # 'upper right')
    ax2.legend(handles=(temp_plot,atmo_plot,diff_plot))

    fig.tight_layout()
    fig.savefig('../tmp/plot_meter.png')
    plt.show()


def main():
    ds_name = 'd2'
    logger.debug(f'data set name={ds_name}')
    chart_meter(*data_sets[ds_name])


if __name__ == '__main__':
    main()
