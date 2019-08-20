# chart
import collections
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from db import Db
from my_utils import Misc


def main():
    ReadingShortRecord = collections.namedtuple('ReadingShortRecord', 'dtime, meter_id, temperature')
    start_dtime = '20190820T182700'
    end_dtime = '30190820T170100'
    readings = Db.select_many('select * '
                              'from readings_short '
                              'where datetime >= ? and datetime <= ?'
                              'order by datetime ',
                              (start_dtime,end_dtime), ReadingShortRecord)
    print(f'{len(readings)} metering have been read')

    dt = [Misc.str_to_dtime(r.dtime) for r in readings if r.meter_id==1]
    temp = [r.temperature for r in readings if r.meter_id==1]
    fig, ax = plt.subplots()
    ax.plot(dt,temp)
    fig.savefig('../tmp/char.png')
    plt.show()

if __name__ == '__main__':
    main()
