# -*- coding: utf-8 -*-
"""
Created: 09/12/2020
1) Read station.smet or station.ext.smet data from a path.

2) Resample and computation of the mean of all the 'smet' features
in a 24h window (from 'validfrom' --> 'validto' dates):
      # validfrom = first recording date of the file (for example: '2020-11-24 00:00:00').
        Date & time of the forecast
      # validto = validto + 1 day (-3h) (for example: '2020-11-24 21:00:00',
        which is the last timestamp of a day (3h sampling)).

3) Computation of three new 'smet' feaures:
      # 'HN24_mean_7d': Sum of the 7 days, 24 hours mean of 'HN24'
      # 'wind_trans24_mean_7d': Sum of the 7 days, 24 hours mean of 'wind_trans24'
      # 'wind_trans24_mean_3d': Sum of the 3 days, 24 hours mean of 'wind_trans24'
@author: Cristina Perez
"""

import numpy as np
import pandas as pd
import re
from pathlib import Path


def read_smet_header(fin):
    if not isinstance(fin, Path):
        fin = Path(fin)
    header_regex = r"^fields           = (.+)$"
    with open(fin, "r") as f:
        nheader = 0
        for line in f:
            match = re.match(header_regex, line.strip())
            if match is not None:
                header = match.group(1).split()
                return header, nheader
            nheader += 1

    raise Exception(
        f"Could not find the header in the file {fin} with regex {header_regex}"
    )


def read_smet(fin):
    if not isinstance(fin, Path):
        fin = Path(fin)
    header, nheader = read_smet_header(fin)
    df = pd.read_csv(
        fin,
        header=None,
        names=header,
        skiprows=nheader + 2,
        sep=r"\s+",
        parse_dates=["timestamp"],
        na_values=["-999"],
    )
    # some columns can have 999 instead of nan or -999, replace with nan
    cols_999 = ["Sd", "Sn", "Ss", "S4", "S5"]
    for col_999 in cols_999:
        df.loc[df[col_999] == 999, col_999] = np.nan
    return df
