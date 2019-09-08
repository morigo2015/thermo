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
status_colorspace = {Status.UNDEF: [0 / 3., 'gray'],
                     Status.GREEN: [1 / 3., 'green'], Status.YELLOW: [2 / 3., 'yellow'], Status.RED: [3 / 3., 'red']}
colorscale = [val for status, val in status_colorspace.items()]

hist_meters_lst = Db.load_hist_meters()  # todo check if (equip_id,dtime) is unique (need for correct pivoting)
hist_equips_lst = Db.select('select * from Hist_equips order by dtime',(),Db.HistEquipsRecord)
hm_df = pd.DataFrame(hist_equips_lst)
hm_df = hm_df[['equip_id','dtime','status_temp','status_atmo','status_group']]

is_an_atmo = hm_df.apply(lambda x: not Db.meter_is_atmo(x.equip_id), axis=1)
hm_df = hm_df.loc[is_an_atmo]

hm_df['status'] = hm_df.apply(
    lambda x: status_colorspace[Informer.aggregate_statuses(x.status_temp, x.status_atmo, x.status_group)][0],
    axis=1)

pivot_table = pd.pivot_table(hm_df[['equip_id', 'dtime', 'status']], index=['equip_id'],
                             columns=['dtime'], values='status', aggfunc='max',
                             fill_value=status_colorspace[Status.UNDEF][0])

meters = [f'місце {m}' for m in pivot_table.index.tolist()]
dates = [Misc.str_time(d) for d in pivot_table.columns.tolist()]
z = pivot_table.values.tolist()

import itertools
zz = itertools.chain(z)

heatmap = go.Heatmap(z=z, x=dates, y=meters, colorscale=colorscale,
                     colorbar=dict(thickness=25,
                                   tickvals=[0.1,0.3,0.6,0.8],
                                   ticktext=['aaa','bbb','ccc','ddd']))
fig = go.Figure(data=heatmap)
fig.show()
fig.write_image("../tmp/tst.png")
