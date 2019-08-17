# readings_full: sqlite -> *.csv -> server(Gdrive)

import csv
import collections

from db import Db
from my_utils import Misc

# sql_fname = '../sql/test'

# Readings_ext = collections.namedtuple('Readings', 'dtime, meter_id, temperature, ser_start, ser_seq')
Readings = collections.namedtuple('Readings', 'dtime, meter_id, temperature')

rds = Db.select_many('select datetime, meter_id, temperature from Readings where meter_id=? order by datetime',
                     (1,), Readings)
rds_ext = []  # [Readings_ext(Misc.str_to_dtime(r[0]), r.meter_id, r.temperature, -1, -1, -1) for r in rds]

for ind in range(len(rds)):
    if ind == 0:
        ser_start = rds[ind].dtime
        ser_seq = 0
    else:
        gap =(Misc.str_to_dtime(rds[ind].dtime) - Misc.str_to_dtime(rds[ind - 1].dtime)).seconds
        print(f'  gap={gap}')
        if gap > 45:
            ser_start = rds[ind].dtime
            ser_seq = 0
        else:
            ser_seq += 1

    rds_ext.append((rds[ind], ser_start, ser_seq))

for r in rds_ext:
    print(r)

with open('../tmp/res.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for r in rds_ext:
        writer.writerow((r[0].dtime, r[0].meter_id, r[0].temperature, r[1], r[2]))
