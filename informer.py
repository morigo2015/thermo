import os

from my_utils import Status
from config import Config
from db import Db, MeterGrpInfo

class Cfg(Config):
    report_fname_path = '../tmp/report.txt'
    report_chart_path = '../tmp/report.png'
    extend_report = False  # if True - include all values not yellow and red only


class Informer:

    @classmethod
    def status_ext_info(cls, status_temp, status_atmo, status_group):
        ext_info = f'(t={Status.status_color_reps[status_temp]},' \
                   f'd={Status.status_color_reps[status_atmo]},' \
                   f'g={Status.status_color_reps[status_group]})'
        return ext_info if Cfg.extend_report else ''

    indicator_names = ['temp', 'atmo_diff']  # , 'group_diff']
    # indicator_signs = ['t', 'd', 'g']

    # @classmethod
    # def get_indic_value(cls, indicator_num, temperatue, atmo_temp, group_temp):
    #     diff_atmo = temperatue - atmo_temp if atmo_temp is not None else None
    #     diff_group = temperatue - group_temp if group_temp is not None else None
    #     return [temperatue, diff_atmo, diff_group][indicator_num]

    @classmethod
    def get_equip_report(cls, equip_hist, meters_hist_lst):
        # generate summary report and alert_report
        if equip_hist.status_atmo is None or equip_hist.status_group is None:
            equip_status = Status.UNDEF
        else:
            equip_status = max(equip_hist.status_temp, equip_hist.status_atmo, equip_hist.status_group)

        ext_info = cls.status_ext_info(equip_hist.status_temp, equip_hist.status_atmo, equip_hist.status_group)
        summary = f'Обладнання {equip_hist.equip_id} - '
        summary += f'{cls.add_markup(Status.status_reps[equip_status], equip_status)}{ext_info}\n'

        if equip_status <= Status.GREEN and not Cfg.extend_report:
            return equip_status, summary, None
        # if equip is not Green then generate alert
        alert = ''
        for m in meters_hist_lst:
            if Db.meter_is_atmo(m.meter_id):
                continue  # no reports for atmo meter
            alert += cls.get_meter_report(m)
        return equip_status, summary, alert

    @classmethod
    def get_meter_report(cls, meter_hist):
        m_stat = max(meter_hist.status_temp, meter_hist.status_atmo, meter_hist.status_group)
        if m_stat <= Status.GREEN and not Cfg.extend_report:
            return None

        temperature = round(meter_hist.temperature, 1) if meter_hist.temperature is not None else None
        atmo_temp = round(meter_hist.atmo_temp, 1) if meter_hist.atmo_temp is not None else None
        group_temp = round(meter_hist.group_temp, 1) if meter_hist.group_temp is not None else None

        # build reference:
        ranges = MeterGrpInfo.get_ranges(meter_hist.meter_id)
        ref = []
        for indicator_num, indic in enumerate(cls.indicator_names):
            yellow_range, red_range = ranges[indicator_num * 2], ranges[indicator_num * 2 + 1]
            if float(yellow_range) == 9999.0 or float(red_range) == 9999.0:
                ref.append(f'[межі не визнач.]')
            else:
                ref.append(f'[межі={yellow_range};{red_range}]')

        ext_info = f'({Status.status_color_reps[meter_hist.status_temp]})' if Cfg.extend_report else ''
        txt = f'  Місце {meter_hist.meter_id}:\n    t={temperature}\'C {ref[0]}{ext_info}\n'
        if atmo_temp is not None:
            txt += f'    t оточення={atmo_temp}\'C\n' \
                   f'    дельта={round(temperature-atmo_temp,1)}\'C {ref[1]}{ext_info}\n'

        return cls.add_markup(txt, m_stat)

    status_markup = {Status.UNDEF: '', Status.GREEN: '', Status.YELLOW: '_', Status.RED: '*'}

    @classmethod
    def add_markup(cls, text, status):
        return cls.status_markup[status] + text + cls.status_markup[status]

    @classmethod
    def inform_user(cls, equip_hist, meters_hist_lst):
        equip_status, summary_report, alert_report = cls.get_equip_report(equip_hist, meters_hist_lst)
        # txt = list(zip(*status_report))[0]  # list of first items (text) of status_report
        # with open(Cfg.report_fname_path, 'a') as f:
        #     f.write(status_report)
        print(summary_report)
        print(alert_report)
        # ColoredText.draw(status_report, Cfg.report_chart_path)
        # os.system(f'telegram-send -i "{Cfg.report_chart_path}"')
        os.system(f'telegram-send --format markdown "{summary_report}"')
        if alert_report:
            os.system(f'telegram-send --format markdown "{alert_report}"')
