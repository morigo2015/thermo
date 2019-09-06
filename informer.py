import os
import logging

from my_utils import Status
from config import Config
from db import Db, MeterGrpInfo
from chart import Chart
from my_utils import Misc

logger = logging.getLogger('thermo.' + 'informer')


class Cfg(Config):
    report_fname_path = '../tmp/report.txt'
    alert_chart_path = '../tmp/alert.png'
    extend_report = False  # if True - include all values not yellow and red only
    config_info = '../configs/thermo_info'
    config_alert = '../configs/thermo_alert'


class Informer:

    @classmethod
    def inform_user(cls, equip_hist, meters_hist_lst):

        equip_status = cls.aggregate_statuses(equip_hist.status_temp, equip_hist.status_temp, equip_hist.status_group)

        # summary line
        summary = f'Обладнання {equip_hist.equip_id} - ' \
                  f'{cls.add_markup(Status.status_reps[equip_status], equip_status)}\n'
        Messenger.send_message(Messenger.SUMMARY_CHANNEL, summary)
        logger.debug(f'summary: {summary}')

        if equip_status <= Status.GREEN and not Cfg.extend_report:
            return equip_status, summary, None
        Messenger.send_message(Messenger.ALERT_CHANNEL, summary)
        for meter_hist in [m for m in meters_hist_lst if not Db.meter_is_atmo(m.meter_id)]:
            alert = cls.get_meter_alert(meter_hist)
            Messenger.send_message(Messenger.ALERT_CHANNEL, alert)
            logger.debug(f'alert for {meter_hist.meter_id}: {alert}')
            cls.make_meter_chart(meter_hist, Cfg.alert_chart_path, alert)
            Messenger.send_chart(Messenger.ALERT_CHANNEL, Cfg.alert_chart_path)

    @classmethod
    def aggregate_statuses(cls, status_temp, status_atmo, status_group):
        if status_atmo is None or status_group is None:
            return Status.UNDEF
        else:
            return max(status_temp, status_atmo, status_group)

    @classmethod
    def get_meter_alert(cls, meter_hist):

        def ref_alert(value, yellow, red):
            if value >= red:
                return f'(>{red}\'C - червона зона)'
            elif value >= yellow:
                return f'(>{yellow}\'C - жовта зона)'
            else:
                logger.error(f'we shouldn\'t be here. ({value},{yellow},{red}')
                return ''

        m_stat = max(meter_hist.status_temp, meter_hist.status_atmo, meter_hist.status_group)
        if m_stat <= Status.GREEN and not Cfg.extend_report:
            return None
        temperature = round(meter_hist.temperature, 1) if meter_hist.temperature is not None else None
        if temperature is None:
            logger.error(f'temperature is None for {meter_hist}')
            return ''
        atmo_temp = round(meter_hist.atmo_temp, 1) if meter_hist.atmo_temp is not None else None
        group_temp = round(meter_hist.group_temp, 1) if meter_hist.group_temp is not None else None

        temp_yellow, temp_red, atmo_yellow, atmo_red, _, _ = MeterGrpInfo.get_ranges(meter_hist.meter_id)
        txt = f'  Точка контролю №{meter_hist.meter_id}:\n'

        if temperature >= temp_yellow:
            txt += f'      t = {temperature}\'C{ref_alert(temperature,temp_yellow,temp_red)}\n'

        atmo_diff = round(temperature - atmo_temp,1) if atmo_temp is not None else None
        if atmo_diff and atmo_diff >= atmo_yellow:
            txt += f'      різниця з оточ. = {atmo_diff}\'C{ref_alert(atmo_diff,atmo_yellow,atmo_red)}\n'

        return cls.add_markup(txt, m_stat)

    @classmethod
    def make_meter_chart(cls, meter_hist, chart_fname_path, alert):
        meter_id = meter_hist.meter_id
        start_dtime = end_dtime = None  # load all history now
        hist_meters_lst = Db.load_hist_meters(meter_id, start_dtime, end_dtime)
        dtime_lst, temperature_lst, atmo_lst = list(zip(*
                                                        [(Misc.str_to_dtime(m.dtime), m.temperature, m.atmo_temp)
                                                         for m in hist_meters_lst if m.atmo_temp is not None]))
        # todo I've just skipped items where atmo is None now. In future it should be processed more carefully.
        delta_atmo_lst = [t - a for t, a in zip(temperature_lst, atmo_lst)]

        ranges = MeterGrpInfo.get_ranges(meter_id)
        Chart.meter_chart([meter_id], dtime_lst, temperature_lst, atmo_lst, delta_atmo_lst, chart_fname_path,
                          ranges=ranges)
        return

    @classmethod
    def add_markup(cls, text, status):
        status_markup = {Status.UNDEF: '', Status.GREEN: '', Status.YELLOW: '_', Status.RED: '*'}
        return status_markup[status] + text + status_markup[status]

    @classmethod
    def _status_ext_info(cls, status_temp, status_atmo, status_group):
        ext_info = f'(t={Status.status_color_reps[status_temp]},' \
                   f'd={Status.status_color_reps[status_atmo]},' \
                   f'g={Status.status_color_reps[status_group]})'
        return ext_info if Cfg.extend_report else ''


class Messenger:
    # channels:
    SUMMARY_CHANNEL = 0
    ALERT_CHANNEL = 1

    chan_configs = {SUMMARY_CHANNEL: Cfg.config_info, ALERT_CHANNEL: Cfg.config_alert}

    @classmethod
    def send_message(cls, channel, message):
        os.system(f'telegram-send --config {cls.chan_configs[channel]} --format markdown "{message}"')

    @classmethod
    def send_chart(cls, channel, chart_fname_path):
        os.system(f'telegram-send --config {cls.chan_configs[channel]} -i "{chart_fname_path}"')


if __name__ == "__main__":
    meter_id = 1004
    dtime = '20190826T122622'
    meter_hist_lst = Db.select('select * from Hist_meters where meter_id=? and dtime=?',
                               (meter_id, dtime), Db.HistMetersRecord)
    alert = "Обладнання 10 - Увага\n\
  Місце 1004:\n\
    t=25.7'C [межі=25.0;26.0]\n\
    t оточення=25.1'C\n\
    дельта=0.6'C [межі=3.0;6.0]"
    Informer.make_meter_chart(meter_hist_lst[0], Cfg.alert_chart_path, alert)
