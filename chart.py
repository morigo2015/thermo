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
    'd1': ('d1',(1041,),'20190824T155400', '20190824T165600'),
    'd2': ('d2',(1041,),'20190824T181000', '20190824T181900'),
    'd3': ('d3',(1004,),'20190824T183700', '20190824T185900')
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


def plot1(dt, temper):
    fig, ax = plt.subplots()
    ax.plot(dt, temper)
    # ax.plot(dt,temp)
    fig.savefig('../tmp/plot1.png')
    plt.show()


def plot2(dt, temper):
    # next we'll write a custom formatter
    N = len(dt)
    ind = np.arange(N)  # the evenly spaced plot indices

    def format_date(x, pos=None):
        thisind = np.clip(int(x + 0.5), 0, N - 1)
        res = Misc.dtime_to_csv_str(dt[thisind])
        return res

    fig, ax = plt.subplots()
    # ax.plot(dt, temper)

    # ax = axes[1]
    ax.plot(ind, temper)  # , 'o-')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax.set_title("Custom tick formatter")
    fig.autofmt_xdate()

    fig.savefig('../tmp/plot2.png')
    plt.show()


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
    myFmt = DateFormatter("%m %H:%M:%S")

    color = 'tab:red'
    temper_plot = ax1.plot(dtime, temper, color=color, label='temper')
    ax1.set_xlabel('time')
    ax1.set_ylabel(f'temperature (id={meter_id})', color=color)
    ax1.xaxis.set_major_formatter(myFmt)
    ax1.xaxis.set_major_locator(ticker.LinearLocator(25))
    # ax1.grid(True)
    fig.autofmt_xdate()

    color = 'tab:green'
    ax1.set_ylabel(f'atmo (id={meter_id})', color=color)
    atmo_plot = ax1.plot(dtime, atmo, color=color, label='atmo')
    ax1.legend(loc='upper left')

    color = 'tab:blue'
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'difference to atmo (id={meter_id}) ', color=color)
    ax2.xaxis.set_major_formatter(myFmt)
    ax2.xaxis.set_major_locator(ticker.LinearLocator(25))
    diff_plot = ax2.plot(dtime, delta_atmo, color=color, label='diff')
    # ax1.legend((temper_plot,atmo_plot,diff_plot),('temperature','atmo','diff'),loc=0)
    ax2.legend(loc='upper right')

    # # ax2.tick_params(axis='y', labelcolor=color)
    # fig.tight_layout()
    # fig.savefig('../tmp/plot_meter.png')
    plt.show()


def main():
    # start_dtime = '20190824T152400'  # '20190824T155400'
    # meter_id = 1041  # -1 means all
    # end_dtime = None
    # end_dtime = '20190824T182000' # '20190824T165600'
    # chart_meter(meter_id, start_dtime, end_dtime)
    ds_name = 'd3'
    logger.debug(f'data set name={ds_name}')
    chart_meter(*data_sets[ds_name])

    # readings = load_readings(start_dtime, end_dtime)
    # selected_readings = [r for r in readings if r.meter_id == meter_id or meter_id == -1]
    # dt = [Misc.str_to_dtime(r.dtime) for r in selected_readings]
    #
    # temper = [r.temperature-r.avg_atmo for r in selected_readings]
    # plot1(dt,temper)
    # plot2(dt, temper)


if __name__ == '__main__':
    main()
