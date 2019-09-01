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
import statistics
from enum import IntEnum

from db import Db, MeterGrpInfo
from chart_utils import ColoredText
from config import Config

logger = logging.getLogger('thermo.' + 'analyzer')


class Cfg(Config):
    report_fname_path = '../tmp/report.txt'
    extend_report = True  # if True - include all values not yellow and red only


class Status(IntEnum):
    UNDEF = -1
    GREEN = 0
    YELLOW = 1
    RED = 2


class Analyzer:
    recs = None  # list readings
    equip_recs_dict = None  # dict( equip_id : [related recs from Readings] )

    @classmethod
    def run(cls):
        cls.recs = Db.select("select * from Readings order by dtime_sec", (), Db.ReadingsRecord, empty_ok=True)
        cls.equip_recs_dict = {equip_id: list(recs) for equip_id, recs
                               in itertools.groupby(cls.recs, key=lambda r: Db.meter_to_equip(r.meter_id))}
        if len(cls.equip_recs_dict) > 1:
            logger.warning(f'More than one equip_id in Readings: {cls.equip_recs_dict.keys()}')

        for equip_id, recs in cls.equip_recs_dict.items():
            logger.debug(f'equip_id={equip_id}, meters set={list(set([r.meter_id for r in recs]))}')
            atmo = cls.get_atmo(recs, equip_id)
            readings_hist_lst = cls.make_readings_hist(recs, atmo)
            equip_hist = cls.make_equips_hist(equip_id, readings_hist_lst)
            readings_hist_lst = cls.add_equip_dtime(readings_hist_lst, equip_hist)
            meters_hist_lst = cls.make_meters_hist(readings_hist_lst, equip_hist.dtime, equip_hist.dtime_sec)
            # equip_status = cls.get_equip_status(readings_hist_lst)
            equip_status, status_report = cls.get_status_report(equip_id, equip_hist, meters_hist_lst)
            cls.do_inform_user(equip_status, status_report)
            cls.save_to_hist(equip_hist, meters_hist_lst, readings_hist_lst)
            cls.delete_readings(recs)

    @classmethod
    def get_atmo(cls, recs, equip_id):
        # calculate atmo for list of readings (of the same equip_id)
        atmo_temps = [r.temperature for r in recs if Db.meter_is_atmo(r.meter_id)]
        if not len(atmo_temps):
            logger.warning(f'there is no atmo meter for equip {equip_id} in meters:{[r.meter_id for r in recs]}')
            return None
        return statistics.mean(atmo_temps)

    @staticmethod
    def calc_status(value, yellow_range, red_range):
        if value is None:
            return Status.UNDEF
        elif value > red_range:
            return Status.RED
        elif value > yellow_range:
            return Status.YELLOW
        else:
            return Status.GREEN

    @classmethod
    def get_reading_statuses(cls, meter_id, temp, atmo):
        ranges = MeterGrpInfo.get_ranges(meter_id)
        if not ranges:
            return Status.UNDEF, Status.UNDEF, Status.UNDEF
        temp_yellow, temp_red, atmo_yellow, atmo_red, _, _ = ranges
        return cls.calc_status(temp, temp_yellow, temp_red), \
               cls.calc_status(atmo, atmo_yellow, atmo_red), \
               Status.UNDEF

    @classmethod
    def make_readings_hist(cls, recs, atmo_temp):
        # extend readings by additional info
        # return list of HistReadingsRecords
        group_temp = None  # not implemented yet
        readings_hist_lst = []
        for r in recs:
            status_temp, status_atmo, status_group = cls.get_reading_statuses(r.meter_id, r.temperature, atmo_temp)
            readings_hist_lst.append(Db.HistReadingsRecord(r.dtime, r.dtime_sec, r.meter_id, r.image_id,
                                                           r.temperature, atmo_temp, group_temp,
                                                           status_temp, status_atmo, status_group,
                                                           None, None))  # equip_dtime, equip_dtime_sec to be updated
        logger.debug(f'len(readings_hist_lst={len(readings_hist_lst)})')
        return readings_hist_lst

    @classmethod
    def add_equip_dtime(cls, readings_hist_lst, equip_hist):
        readings_hist_lst = [r._replace(equip_dtime=equip_hist.dtime, equip_dtime_sec=equip_hist.dtime_sec)
                             for r in readings_hist_lst]
        return readings_hist_lst

    @classmethod
    def make_meters_hist(cls, readings_hist_lst, equip_dtime, equip_dtime_sec):
        # group by meters and create list of records for meter_hist
        readings_hist_lst = sorted(readings_hist_lst, key=lambda r: r.meter_id)
        # grouped_by_meter = list(itertools.groupby(readings_hist_lst, key=lambda r: r.meter_id))
        meters_recs_dict = {meter_id: list(recs)
                            for meter_id, recs in itertools.groupby(readings_hist_lst, key=lambda r: r.meter_id)}
        meters_hist_lst = []
        for m_id, recs in meters_recs_dict.items():
            if not len(recs):
                logger.error(f'recs len =0 for m_id={m_id}, len(readings_hist_lst)={len(readings_hist_lst)}')
            temperature = statistics.mean([r.temperature for r in recs])
            if any(r.atmo_temp is None for r in recs):
                atmo_temp = None
            else:
                atmo_temp = statistics.mean([r.atmo_temp for r in recs])
            group_temp = 9999.0  # statistics.mean([r.group_temp for r in recs])
            status_temp = max([r.status_temp for r in recs])
            status_atmo = max([r.status_atmo for r in recs])
            status_group = max([r.status_group for r in recs])
            meters_hist_lst.append(Db.HistMetersRecord(m_id, equip_dtime, equip_dtime_sec,
                                                       temperature, atmo_temp, group_temp,
                                                       status_temp, status_atmo, status_group))
        logger.debug(f'len(meter_hist_lst)={len(meters_hist_lst)}')
        return meters_hist_lst

    @classmethod
    def make_equips_hist(cls, equip_id, readings_hist_lst):
        dtime = min([r.dtime for r in readings_hist_lst])
        dtime_sec = min([r.dtime_sec for r in readings_hist_lst])
        status_temp = max([r.status_temp for r in readings_hist_lst])
        status_atmo = max([r.status_atmo for r in readings_hist_lst])
        status_group = max([r.status_group for r in readings_hist_lst])
        equips_hist = Db.HistEquipsRecord(equip_id, dtime, dtime_sec, status_temp, status_atmo, status_group)
        return equips_hist

    status_reps = {Status.UNDEF: 'Невизначено',
                   Status.GREEN: 'OK', Status.YELLOW: 'Увага', Status.RED: 'Небезепечно!'}
    status_colors = {Status.UNDEF: ColoredText.undef,
                     Status.GREEN: ColoredText.ok, Status.YELLOW: ColoredText.warning, Status.RED: ColoredText.critical}
    indicator_names = ['temp', 'atmo_diff'] # , 'group_diff']
    indicator_signs = ['t', 'd', 'g']

    @classmethod
    def get_indic_value(cls, indicator_num, temperatue, atmo_temp, group_temp):
        diff_atmo = temperatue - atmo_temp if atmo_temp is not None else None
        diff_group = temperatue - group_temp if group_temp is not None else None
        return [temperatue, diff_atmo, diff_group][indicator_num]

    @classmethod
    def get_status_report(cls, equip_id, equip_hist, meters_hist_lst):  # todo
        # generate status report
        equip_status = max(equip_hist.status_temp, equip_hist.status_atmo, equip_hist.status_group)
        status_report = [(f'Обладнання {equip_id} - {cls.status_reps[equip_status]}',
                          cls.status_colors[equip_status])]
        if equip_status <= Status.GREEN and not Cfg.extend_report:
            return equip_status, status_report
        # if equip is not Green then print all non-green indicator for all meters
        for i, indicator in enumerate(cls.indicator_names):
            eqp_stat = [equip_hist.status_temp, equip_hist.status_atmo, equip_hist.status_group][i]
            if eqp_stat <= Status.GREEN and not Cfg.extend_report:
                continue
            for m in meters_hist_lst:
                m_stat = [m.status_temp, m.status_atmo, m.status_group][i]
                if m_stat < Status.GREEN and not Cfg.extend_report:
                    continue
                status_report.append((cls.get_meter_report(m, i), cls.status_colors[m_stat]))
        return equip_status, status_report

    @classmethod
    def get_meter_report(cls, meter_hist, indicator_num):
        indic_sign = cls.indicator_signs[indicator_num]
        ranges = MeterGrpInfo.get_ranges(meter_hist.meter_id)
        yellow_range, red_range = ranges[indicator_num * 2], ranges[indicator_num * 2 + 1]
        indic_value = cls.get_indic_value(indicator_num,
                                          meter_hist.temperature, meter_hist.atmo_temp, meter_hist.group_temp)
        report = f'\t{indic_sign}(id={meter_hist.meter_id})' \
                 f'={round(indic_value,1) if indic_value is not None else None}\'C '\
                 f'ref:{yellow_range}\'C;{red_range}\'C'
        return report

    @classmethod
    def do_inform_user(cls, equip_status, status_report):
        txt = list(zip(*status_report))[0]  # list of first items (text) of status_report
        with open(Cfg.report_fname_path, 'a') as f:
            for l in txt:
                f.write(l + '\n')
                print(l)

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
            cnt = Db.exec('insert into _Readings_arch select * from Readings where dtime_sec = ? and meter_id = ?',
                          (r.dtime_sec, r.meter_id))
            # logger.debug(f'moved {cnt} records from Readings to _Readings_arch')
            rec_cnt += Db.exec('delete from Readings where dtime_sec = ? and meter_id = ?', (r.dtime_sec, r.meter_id))
            # if cnt != 1:
            #     logger.warning(f'delete Readings failed, cnt={cnt} for {(r.dtime_sec,r.meter_id)}')
        logger.debug(f'deleted {rec_cnt} from Readings')
        return
