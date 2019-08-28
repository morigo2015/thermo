# analyzer.py
# analyze readings related to last equip_id and inform user if need

import logging

from db import Db

logger = logging.getLogger('thermo.' + 'analyzer')


class Analyzer:

    @classmethod
    def run(cls, event):
        recs = Db.select("select * from Readings order by dtime_sec", (), Db.ReadingsRecord, empty_ok=True)
        met_equip_lst = list(set([(r.meter_id, Db.meter_to_equip(r.meter_id)) for r in recs]))
        logger.debug(f'Run:: {len(recs)} readings.  (meter,equip):{met_equip_lst} '
                     f'last meter,equip={recs[-1].meter_id},{Db.meter_to_equip(recs[-1].meter_id)}')

        if event == 'readings_taken' and len(recs) > 1:
            cnt = Db.exec("delete from Readings where dtime_sec <> ?", (recs[-1].dtime_sec,))
        else:
            cnt = Db.exec("delete from Readings")
        logger.debug(f'event={event} deleted {cnt} records from Readings')
