# table Readings --> table Readings_processed, Readings_archive
# actions:
#   - group sequential metering with the same meter_id to (min,avg,max)
#   - fill atmo_temper field for not atmo readings
#   - move Readings --> Readings_archive with reference to Readings_processed.datetime
import logging
import collections
import itertools
from statistics import mean
from my_utils import Misc
from db import Db

logger = logging.getLogger('thermo.' + 'proc_reads')


class ProcessReadings:

    @classmethod
    def do_process(cls):
        readings = Db.select_many('select * from readings_full order by dtime',
                                  (), Db.ReadingsFullRecord, empty_ok=True)
        if not len(readings):
            logger.debug('Readings table is empty')
            return
        cnt = {'readings':len(readings),'archived':0, 'deleted':0, 'groups':0}
        logger.debug(f'load {cnt["readings"]} readings. '
                     f'dtime range: {min([r.dtime for r in readings])} -- {max([r.dtime for r in readings])}')

        # group reads by sequences where equip_id is the same
        equip_grp_reads = [(k, list(g)) for k, g in itertools.groupby(readings, key=lambda x: x.equip_id)]
        for equip_id, equip_grp in equip_grp_reads:
            # for each such sequence (where equip_id=const) group by meter_id:
            equip_grp_sorted = sorted(list(equip_grp), key=lambda x: x.meter_id)
            meter_grp_reads = [(meter_id, list(meter_grp)) for meter_id, meter_grp
                               in itertools.groupby(equip_grp_sorted, key=lambda x: x.meter_id)]
            for meter_id, meter_grp in meter_grp_reads:
                processed_dtime = min([r.dtime for r in meter_grp])  # time of first reading in group
                temper_lst = [r.temperature for r in meter_grp]  # equip_id -- meter_id
                min_temper = min(temper_lst)
                avg_temper = mean(temper_lst)
                max_temper = max(temper_lst)
                readings_grouped = Db.ReadingsGroupedRecord(
                    processed_dtime, meter_id, min_temper, avg_temper, max_temper, None, None)
                Db.insert_one('Readings_grouped', readings_grouped)
                cnt['groups'] += 1

                # move record from Readings to ReadingsArchive table
                for r in meter_grp:
                    readings_archived = Db.ReadingsArchivedRecord(
                        r.dtime, r.meter_id, r.image_id, r.temperature, processed_dtime)
                    Db.insert_one('Readings_archived', readings_archived)
                    cnt['archived'] += 1
                    rc = Db.exec(f'delete from Readings where dtime=? and meter_id=?', (r.dtime, r.meter_id))
                    if rc != 1:
                        logger.error(
                            f'something wrong with deleting Readings(dtime={r.dtime},meter_id={r.meter_id}  rc={rc}')
                    cnt['deleted'] += rc
        if cnt['readings'] != cnt['archived'] or cnt['readings'] != cnt['deleted']:
            logger.warning(f'Something wrong with counters: {cnt}')
        else:
            logger.info(f'{cnt["readings"]} records moved from Readings to Readings_archive. '
                        f'{cnt["groups"]} records inserted in Readings_grouped')

def main():
    ProcessReadings.do_process()


if __name__ == "__main__":
    main()
