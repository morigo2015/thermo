# table Readings --> table Readings_processed, Readings_archive
# actions:
#   - group sequential metering with the same meter_id to (min,avg,max)
#   - fill atmo_temper field for not atmo readings
#   - move Readings --> Readings_archive with reference to Readings_processed.datetime
import logging
import itertools
from statistics import mean
from my_utils import Misc
from db import Db

logger = logging.getLogger('thermo.' + 'proc_reads')


class ProcessReadings:

    @classmethod
    def do_process(cls):
        readings = Db.select('select * from readings_full order by dtime',
                             (), Db.ReadingsFullRecord, empty_ok=True)
        if not len(readings):
            logger.debug('Readings table is empty')
            return
        logger.debug(f'loaded {len(readings)} readings. '
                     f'dtime range: {min([r.dtime for r in readings])} -- {max([r.dtime for r in readings])}')

        reads_grp_lst = cls._group_by_equip_meter(readings)  # group by seq equip_id then by meter_id
        # reads_grp_lst = cls._update_atmo_temper(reads_grp_lst)
        cls._move_reads(readings, reads_grp_lst)  # move records: Readings --> Readings_grouped,Readings_archive

    @classmethod
    def _group_by_equip_meter(cls, readings):
        # group reads by sequences where equip_id is the same
        reads_grp_lst = []
        reads_arch_lst = []
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
                reads_grp_lst.append(Db.ReadingsGroupedRecord(
                    processed_dtime, Misc.dtime_to_sqlite_str(Misc.str_to_dtime(processed_dtime)), meter_id,
                    min_temper, avg_temper, max_temper, None, None))
                reads_arch_lst.append(Db.ReadingsArchivedRecord(@@@@))

        logger.debug(f'created {len(reads_grp_lst)} groups in reads_grp_lst')
        return reads_grp_lst

    @classmethod
    def _update_atmo_temper(cls, r_grp_lst):
        pass

    @classmethod
    def _move_reads(cls, readings, readings_grouped):
        cnt_ins = cnt_del = 0
        for r_grp in readings_grouped:
            Db.insert_one('Readings_grouped', r_grp)
            for r in readings:
                if not (r_grp.equip_id == r.equip_id and r_grp.meter_id == r.meter_id):
                    continue  # skip if r was not included in r_grp
                Db.insert_one('Readings_archived', Db.ReadingsArchivedRecord(
                    r.dtime, r.meter_id, r.image_id, r.temperature, r_grp.dtime))
                cnt_ins += 1

                rc = Db.exec(f'delete from Readings where dtime=? and meter_id=?', (r.dtime, r.meter_id))
                if rc != 1:
                    logger.error(
                        f'something is wrong while deleting Readings({r.dtime},meter_id={r.meter_id} rc={rc}')
                cnt_del += rc

        if cnt_ins != len(readings) or cnt_del != len(readings):
            logger.warning(f'Wrong counters: len(readings)={len(readings)}, inserted={cnt_ins}, deleted={cnt_del}')
        else:
            logger.info(f'Counters - OK:  {len(readings)} records moved from Readings to Readings_archive. ')


def main():
    ProcessReadings.do_process()


if __name__ == "__main__":
    main()
