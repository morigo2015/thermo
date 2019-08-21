# chart
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


from db import Db
from my_utils import Misc

ReadingShortRecord = collections.namedtuple('ReadingShortRecord', 'dtime, meter_id, temperature')

def load_readings(start_dtime=None, end_dtime=None):
    if not start_dtime and not end_dtime:
        where_clause, query_args = '',()
    elif start_dtime and not end_dtime:
        where_clause, query_args = 'where datetime >= ?', (start_dtime,)
    elif not start_dtime and end_dtime:
        where_clause, query_args = 'where datetime <= ?', (end_dtime,)
    elif start_dtime and end_dtime:
        where_clause, query_args = 'where datetime >= ? and datetime <=?', (start_dtime,end_dtime)
    else:
        print(f'are you kidding me? start={start_dtime} end={end_dtime}')
        exit(1)

    query = f'select * from readings_short {where_clause} order by datetime'
    readings = Db.select_many( query, query_args, ReadingShortRecord)
    print(f'loaded {len(readings)} readings, '
          f'dtime range: {min([r.dtime for r in readings])} : {max([r.dtime for r in readings])}')
    return readings

def plot1(dt,temper):
    fig, ax = plt.subplots()
    ax.plot(dt, temper)
    # ax.plot(dt,temp)
    fig.savefig('../tmp/plot1.png')
    plt.show()

def plot2(dt,temper):
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
    ax.plot(ind, temper) # , 'o-')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
    ax.set_title("Custom tick formatter")
    fig.autofmt_xdate()

    fig.savefig('../tmp/plot2.png')
    plt.show()


def main():
    start_dtime = '20190821T130800'
    meter_id = 1  # -1 means all
    conv_to_time = 1
    end_dtime = None
    # end_dtime = '20190821T154700'
    readings = load_readings(start_dtime)

    selected_readings = [r for r in readings if r.meter_id==meter_id or meter_id==-1]
    dt = [Misc.str_to_dtime(r.dtime) if conv_to_time else r.dtime for r in selected_readings]
    temper = [r.temperature for r in selected_readings]
    # plot1(dt,temper)
    plot2(dt,temper)

if __name__ == '__main__':
    main()
