import datetime
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly
import numpy as np
import pandas as pd

from db import Db
from informer import Informer
from my_utils import Status, Misc

plotly.io.orca.config.executable = '/home/im/Downloads/orca-1.2.1-x86_64.AppImage'
hist_equips_lst = Db.select('select * from Hist_equips order by dtime', (), Db.HistEquipsRecord)
hm_df = pd.DataFrame(hist_equips_lst)
hm_df = hm_df[['equip_id', 'dtime', 'status_temp', 'status_atmo', 'status_group', 'cycle_dtime']]

scaled_status = {Status.UNDEF: 0., Status.GREEN: 0.3, Status.YELLOW: 0.6, Status.RED: 1.}
# colorscale = [val for status, val in status_colorspace.items()]
colorscale = [[0., 'gray'], [0.2, 'green'], [0.5, 'yellow'], [1., 'red']]
# colorscale = [[0., 'gray'], [0.2, 'gray'],
#               [0.2, 'green'], [0.5, 'green'],
#               [0.5, 'yellow'], [0.7, 'yellow'],
#               [0.7, 'red'], [1.0, 'red']]

hm_df['status'] = hm_df.apply(
    lambda x: Informer.aggregate_statuses(x.status_temp, x.status_atmo, x.status_group), axis=1)
hm_df['scaled_status'] = hm_df.apply(lambda x: scaled_status[x.status], axis=1)

pivot_table = pd.pivot_table(hm_df[['equip_id', 'dtime', 'status', 'cycle_dtime', 'status', 'scaled_status']],
                             index=['equip_id'],
                             columns=['cycle_dtime'], values='scaled_status',
                             aggfunc='max',
                             fill_value=scaled_status[Status.UNDEF])

equips = [f'місце {m}' for m in pivot_table.index.tolist()]
dates = [Misc.str_time(d) for d in pivot_table.columns.tolist()]
z = pivot_table.values.tolist()

heatmap = go.Heatmap(z=z, x=dates, y=equips, colorscale=colorscale, xgap=2, ygap=2)
fig = go.Figure(data=[heatmap])
# fig = ff.create_annotated_heatmap(z=z, x=dates, y=equips, colorscale=colorscale, xgap=2, ygap=2)
fig.show()
fig.write_image("../tmp/tst.png")

# colorbar=dict(thickness=25,
#               tickvals=[0.1, 0.4, 0.7, 0.9],
#               ticktext=['aaa', 'bbb', 'ccc', 'ddd']))
