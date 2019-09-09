# analyzer.py
# analyze readings related to last equip_id and inform user if need:
# for each group by equip_id (excluding last record if need):
# - calculate atmo
# - for each reading extend reading info: add atmo; status_temper, status_atmo
# - get equip status
# - inform user
# - add records to Readings_hist, remove from Readings
#
# general pipeline:
# flir_file --> several records in Readings: one per meter_id in image
# for all readings related to sequential equip_id:
#   1 rec in Readings --> 1 rec in Readings_hist: extending by atmo and statuses for temp,atmo,group
#   all recs in Readings_hist --> 1 rec in Equip_hist: where dtime=min(dtime), status=worst(status)
#   each rec in Reading_hist update: equip_dtime = equip_hist.dtime, to collect all readings related to equip
#   group by meter_id (inside the same equip_id):  (for future reports mostly)
#     each group(meter_id) --> 1 rec Hist_meters: dtime=min(dtime), temp=mean(temp), status=worst(status)

import logging
import itertools
from statistics import mean

from db import Db, MeterGrpInfo
from my_utils import Status
from informer import Informer
from reporter import Reporter

logger = logging.getLogger('thermo.' + 'analyzer')


class Analyzer:
    recs = None  # list readings
    equip_recs_dict = None  # dict( equip_id : [related recs from Readings] )

    @classmethod
    def run(cls):
        cls.recs = Db.select("select * from Readings order by dtime_sec", (), Db.ReadingsRecord, empty_ok=True)

        # group readings by meter_id:
        cls.equip_recs_dict = {equip_id: list(recs) for equip_id, recs
                               in itertools.groupby(cls.recs, key=lambda r: Db.meter_to_equip(r.meter_id))}
        if len(cls.equip_recs_dict) > 1:
            logger.warning(f'More than one equip_id in Readings: {cls.equip_recs_dict.keys()}')

        for equip_id, recs in cls.equip_recs_dict.items():
            logger.debug(f'equip_id={equip_id}, equips set={list(set([r.meter_id for r in recs]))}')

            readings_hist_lst, equip_dtime, equip_dtime_sec = cls.make_readings_hist(recs)
            atmo = cls.get_atmo(recs, equip_id)  # atmo = average of all atmo's for this equip_id
            meters_hist_lst = cls.make_meters_hist(readings_hist_lst, equip_dtime, equip_dtime_sec, atmo)
            equip_hist, new_cycle_started_flg = cls.make_equips_hist(equip_id, meters_hist_lst,
                                                                     equip_dtime, equip_dtime_sec)
            # move to *_hist from Readings.
            # Do it before inform_user which loads all values for alert_chart from Hit_meters
            cls.save_to_hist(equip_hist, meters_hist_lst, readings_hist_lst)
            cls.delete_readings(recs)

            Informer.inform_user(equip_hist, meters_hist_lst)
            if new_cycle_started_flg:
                Reporter.send_equip_heatmap()

    @classmethod
    def make_readings_hist(cls, recs):
        # extend readings by additional info
        # return list of HistReadingsRecords
        readings_hist_lst = []
        for r in recs:
            # status_temp, status_atmo, status_group = cls.get_reading_statuses(r.meter_id, r.temperature, atmo_temp)
            readings_hist_lst.append(Db.HistReadingsRecord(r.dtime, r.dtime_sec, r.meter_id,
                                                           r.image_id, r.temperature,
                                                           None, None))  # equip_dtime, equip_dtime_sec to be updated
        equip_dtime = min([r.dtime for r in readings_hist_lst])
        equip_dtime_sec = min([r.dtime_sec for r in readings_hist_lst])
        logger.debug(f'len(readings_hist_lst={len(readings_hist_lst)})')
        return readings_hist_lst, equip_dtime, equip_dtime_sec

    @classmethod
    def get_atmo(cls, recs, equip_id):
        # calculate atmo for list of readings (of the same equip_id)
        atmo_temps = [r.temperature for r in recs if Db.meter_is_atmo(r.meter_id)]
        if not len(atmo_temps):
            logger.warning(f'there is no atmo meter for equip {equip_id} in equips:{[r.meter_id for r in recs]}')
            return None
        return mean(atmo_temps)

    @classmethod
    def get_reading_statuses(cls, meter_id, temp, atmo):
        ranges = MeterGrpInfo.get_ranges(meter_id)
        if not ranges:
            return Status.UNDEF, Status.UNDEF, Status.UNDEF
        temp_yellow, temp_red, atmo_yellow, atmo_red, _, _ = ranges
        temp_status = Status.calc_status(temp, temp_yellow, temp_red)
        atmo_status = Status.calc_status(temp - atmo, atmo_yellow, atmo_red) if atmo is not None else Status.UNDEF
        group_status = Status.UNDEF
        return temp_status, atmo_status, group_status

    @classmethod
    def make_meters_hist(cls, readings_hist_lst, equip_dtime, equip_dtime_sec, atmo_temp):
        # group by equips and create list of records for meter_hist
        readings_hist_lst = sorted(readings_hist_lst, key=lambda r: r.meter_id)
        meters_recs_dict = {meter_id: list(recs)
                            for meter_id, recs in itertools.groupby(readings_hist_lst, key=lambda r: r.meter_id)}
        meters_hist_lst = []
        for m_id, recs in meters_recs_dict.items():
            if not len(recs):
                logger.error(f'recs len =0 for m_id={m_id}, len(readings_hist_lst)={len(readings_hist_lst)}')
            temperature = mean([r.temperature for r in recs])
            group_temp = None
            status_temp, status_atmo, status_group = cls.get_reading_statuses(m_id, temperature, atmo_temp)
            meters_hist_lst.append(Db.HistMetersRecord(m_id, equip_dtime, equip_dtime_sec,
                                                       temperature, atmo_temp, group_temp,
                                                       status_temp, status_atmo, status_group))
        logger.debug(f'len(meter_hist_lst)={len(meters_hist_lst)}')
        return meters_hist_lst

    @classmethod
    def make_equips_hist(cls, equip_id, meters_hist_lst, equip_dtime, equip_dtime_sec):
        status_temp = max([r.status_temp for r in meters_hist_lst])
        status_atmo = max([r.status_atmo for r in meters_hist_lst])
        status_group = max([r.status_group for r in meters_hist_lst])
        cycle_dtime, new_cycle_started_flg = cls.get_cycle_dtime(equip_id, equip_dtime)
        equips_hist = Db.HistEquipsRecord(equip_id, equip_dtime, equip_dtime_sec,
                                          status_temp, status_atmo, status_group, cycle_dtime)
        return equips_hist, new_cycle_started_flg

    @classmethod
    def get_cycle_dtime(cls, equip_id, equip_dtime) -> (str,bool):
        #  return: (current cycle start_dtime, new_cycle_started_flag)
        # cycle is a sequential records in Hist_equip where no equip_id is repeating
        # as soon as equip_id is repeated - we start new cycle.
        # cycle_dtime is dtime when current cycle (related to the record in Hist_equips) started
        max_cycle_dtime = Db.select('select max(cycle_dtime) from Hist_equips',
                                    (), Db.OneValueRecord, empty_ok=True)[0].value
        if max_cycle_dtime is None:  # empty table
            return equip_dtime, True
        equip_lst = [r.equip_id for r in Db.select('select * from Hist_equips where cycle_dtime = ?',
                                                   (max_cycle_dtime,), Db.HistEquipsRecord, empty_ok=True)]
        if not len(equip_lst):
            logger.error(f'equip_lst is empty. equip_id={equip_id} max_cycle_dtime={max_cycle_dtime}')
            return equip_dtime, True

        if equip_id not in equip_lst:  # cycle is still continuing
            return max_cycle_dtime, False
        else:  # equip_id repeated - start new cycle
            return equip_dtime, True

    @classmethod
    def save_to_hist(cls, equip_hist, meters_hist_lst, readings_hist_lst):
        logger.debug(f'equip({equip_hist.equip_id},{equip_hist.dtime}, '
                     f'meters_len={len(meters_hist_lst)} readings_len={len(readings_hist_lst)}')
        Db.insert_one('Hist_equips', equip_hist)
        Db.insert_many('Hist_meters', meters_hist_lst)
        Db.insert_many('Hist_readings', readings_hist_lst)

    @classmethod
    def delete_readings(cls, recs):
        logger.debug(f'recs: len={len(recs)}')
        rec_cnt = 0
        for r in recs:
            Db.exec('insert into _Readings_arch select * from Readings where dtime_sec = ? and meter_id = ?',
                    (r.dtime_sec, r.meter_id))
            # logger.debug(f'moved {cnt} records from Readings to _Readings_arch')
            rec_cnt += Db.exec('delete from Readings where dtime_sec = ? and meter_id = ?',
                               (r.dtime_sec, r.meter_id))
        logger.debug(f'deleted {rec_cnt} from Readings')
        return
