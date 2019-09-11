import pandas as pd
import logging

from chart import Chart
from my_utils import Misc, Status
from db import Db
from informer import Informer, Messenger
from config import Config

logger = logging.getLogger('thermo.' + 'reporter')


class Cfg(Config):
    heatmap_fname_path = '../tmp/heatmap.png'


class Reporter:

    @classmethod
    def send_equip_heatmap(cls):
        hist_equips_lst = Db.select('select * from Hist_equips order by dtime', (), Db.HistEquipsRecord)
        hm_df = pd.DataFrame(hist_equips_lst)
        if len(hm_df) < 2:
            logger.debug(f'not enough values in Hist_equips. len={len(hm_df)}')
            return
        # delete last record. it relates to new (uncompleted cycle) and should not be in chart
        hm_df = hm_df.drop(index=max(hm_df.index), axis=0)

        hm_df = hm_df[['equip_id', 'dtime', 'status_temp', 'status_atmo', 'status_group', 'cycle_dtime']]

        hm_df['status'] = hm_df.apply(
            lambda x: Informer.aggregate_statuses(x.status_temp, x.status_atmo, x.status_group), axis=1)

        pivot_table = pd.pivot(hm_df, index='equip_id', columns='cycle_dtime', values='status')
        pivot_table = pivot_table.fillna(Status.UNDEF)

        equips = [f'Двигун №{m}' for m in pivot_table.index.tolist()]
        dates = [Misc.str_readable(d) for d in pivot_table.columns.tolist()]
        # z = pivot_table.values.tolist()
        statuses = pivot_table.to_numpy()

        Chart.equip_heatmap(statuses, equips, dates, Cfg.heatmap_fname_path, show=False)

        Messenger.send_chart(Messenger.REPORT_CHANNEL, Cfg.heatmap_fname_path)


if __name__ == "__main__":
    Reporter.send_equip_heatmap()
