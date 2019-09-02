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
from enum import IntEnum
import os

from db import Db, MeterGrpInfo
from chart_utils import ColoredText
from config import Config

logger = logging.getLogger('thermo.' + 'analyzer')


class Cfg(Config):
    report_fname_path = '../tmp/report.txt'
    report_chart_path = '../tmp/report.png'
    extend_report = False  # if True - include all values not yellow and red only


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

        # group readings by meter_id:
        cls.equip_recs_dict = {equip_id: list(recs) for equip_id, recs
                               in itertools.groupby(cls.recs, key=lambda r: Db.meter_to_equip(r.meter_id))}
        if len(cls.equip_recs_dict) > 1:
            logger.warning(f'More than one equip_id in Readings: {cls.equip_recs_dict.keys()}')

        for equip_id, recs in cls.equip_recs_dict.items():
            logger.debug(f'equip_id={equip_id}, meters set={list(set([r.meter_id for r in recs]))}')

            readings_hist_lst, equip_dtime, equip_dtime_sec = cls.make_readings_hist(recs)
            atmo = cls.get_atmo(recs, equip_id)  # atmo = average of all atmo's for this equip_id
            meters_hist_lst = cls.make_meters_hist(readings_hist_lst, equip_dtime, equip_dtime_sec, atmo)
            equip_hist = cls.make_equips_hist(equip_id, meters_hist_lst, equip_dtime, equip_dtime_sec)

            equip_status, status_report = cls.get_equip_report(equip_id, equip_hist, meters_hist_lst)
            cls.do_inform_user(equip_status, status_report)

            cls.save_to_hist(equip_hist, meters_hist_lst, readings_hist_lst)
            cls.delete_readings(recs)

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
            logger.warning(f'there is no atmo meter for equip {equip_id} in meters:{[r.meter_id for r in recs]}')
            return None
        return mean(atmo_temps)

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
        temp_status = cls.calc_status(temp, temp_yellow, temp_red)
        atmo_status = cls.calc_status(temp - atmo, atmo_yellow, atmo_red) if atmo is not None else Status.UNDEF
        group_status = Status.UNDEF
        return temp_status, atmo_status, group_status

    status_reps = {Status.UNDEF: 'Невизначено',
                   Status.GREEN: 'OK', Status.YELLOW: 'Увага', Status.RED: 'Небезепечно!'}
    status_color_reps = {Status.UNDEF: 'Undef',
                         Status.GREEN: 'Green', Status.YELLOW: 'Yellow', Status.RED: 'Red'}
    status_colors = {Status.UNDEF: ColoredText.undef,
                     Status.GREEN: ColoredText.ok, Status.YELLOW: ColoredText.warning, Status.RED: ColoredText.critical}

    @classmethod
    def make_meters_hist(cls, readings_hist_lst, equip_dtime, equip_dtime_sec, atmo_temp):
        # group by meters and create list of records for meter_hist
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
        equips_hist = Db.HistEquipsRecord(equip_id, equip_dtime, equip_dtime_sec,
                                          status_temp, status_atmo, status_group)
        return equips_hist

    indicator_names = ['temp', 'atmo_diff']  # , 'group_diff']
    indicator_signs = ['t', 'd', 'g']

    @classmethod
    def get_indic_value(cls, indicator_num, temperatue, atmo_temp, group_temp):
        diff_atmo = temperatue - atmo_temp if atmo_temp is not None else None
        diff_group = temperatue - group_temp if group_temp is not None else None
        return [temperatue, diff_atmo, diff_group][indicator_num]

    @classmethod
    def status_ext_info(cls, status_temp, status_atmo, status_group):
        ext_info = f'(t={cls.status_color_reps[status_temp]},' \
                   f'd={cls.status_color_reps[status_atmo]},' \
                   f'g={cls.status_color_reps[status_group]})'
        return ext_info if Cfg.extend_report else ''

    @classmethod
    def get_equip_report(cls, equip_id, equip_hist, meters_hist_lst):  # todo
        # generate status report
        if equip_hist.status_atmo is None or equip_hist.status_group is None:
            equip_status = Status.UNDEF
        else:
            equip_status = max(equip_hist.status_temp, equip_hist.status_atmo, equip_hist.status_group)

        ext_info = cls.status_ext_info(equip_hist.status_temp, equip_hist.status_atmo, equip_hist.status_group)
        status_report = [(f'Обладнання {equip_id} - {cls.status_reps[equip_status]}{ext_info}',
                          cls.status_colors[equip_status])]

        if equip_status <= Status.GREEN and not Cfg.extend_report:
            return equip_status, status_report
        # if equip is not Green then print all non-green indicator for all meters
        for m in meters_hist_lst:
            if Db.meter_is_atmo(m.meter_id):
                continue  # no reports for atmo meter
            meter_report = cls.get_meter_report(m)
            if meter_report is not None:
                status_report.append(meter_report)
            # for i, indicator in enumerate(cls.indicator_names):
            #     m_stat = [m.status_temp, m.status_atmo, m.status_group][i]
            #     # eqp_stat = [equip_hist.status_temp, equip_hist.status_atmo, equip_hist.status_group][i]
            #     # if eqp_stat <= Status.GREEN and not Cfg.extend_report:
            #     #     continue
            #     status_report.append((cls.get_meter_report(m, i), cls.status_colors[m_stat]))
        return equip_status, status_report

    @classmethod
    def get_meter_report(cls, meter_hist):
        m_stat = max(meter_hist.status_temp, meter_hist.status_atmo, meter_hist.status_group)
        if m_stat <= Status.GREEN and not Cfg.extend_report:
            return None

        temperature = round(meter_hist.temperature,2) if meter_hist.temperature is not None else None
        atmo_temp = round(meter_hist.atmo_temp,2) if meter_hist.atmo_temp is not None else None
        group_temp = round(meter_hist.group_temp,2) if meter_hist.group_temp is not None else None

        ranges = MeterGrpInfo.get_ranges(meter_hist.meter_id)
        ref = []
        for indicator_num, indic in enumerate(cls.indicator_names):
            yellow_range, red_range = ranges[indicator_num * 2], ranges[indicator_num * 2 + 1]
            if float(yellow_range) == 9999.0 or float(red_range) == 9999.0:
                ref.append(f'[межі не визнач.]')
            else:
                ref.append(f'[межі={yellow_range};{red_range}]')

        ext_info = f'({cls.status_color_reps[meter_hist.status_temp]})' if Cfg.extend_report else ''
        report_txt = f'  Місце {meter_hist.meter_id}: ' \
                 f't={temperature}{ref[0]}{ext_info}'
        if atmo_temp is not None:
            report_txt += f', t оточення={atmo_temp},різниця={round(temperature-atmo_temp,2)}' \
                      f'{ref[1]}{ext_info}'

        return report_txt, cls.status_colors[m_stat]

    @classmethod
    def do_inform_user(cls, equip_status, status_report):
        txt = list(zip(*status_report))[0]  # list of first items (text) of status_report
        with open(Cfg.report_fname_path, 'a') as f:
            for l in txt:
                f.write(l + '\n')
                print(l)
        ColoredText.draw(status_report, Cfg.report_chart_path)
        os.system(f'telegram-send -i "{Cfg.report_chart_path}"')

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
