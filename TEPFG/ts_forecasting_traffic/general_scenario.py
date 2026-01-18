import os
import pandas as pd
import numpy as np

from ts_forecasting_traffic.scenario import ScenarioAbstract

import zipfile

def time_add(data, week_start, interval=5, weekday_only=False, holiday_list=None, day_start=0, hour_of_day=24):
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    holiday_data = np.zeros_like(data)

    day_init = day_start
    week_init = week_start
    holiday_init = 1

    for index in range(day_start // interval, data.shape[0] + day_start // interval):
        if (index) % time_slot == 0 and index != 0:
            day_init = 0
        day_init = day_init + interval
        if (index) % time_slot == 0 and index != 0:
            week_init = week_init + 1
        if week_init > week_max:
            week_init = 1
        if day_init < 6:
            holiday_init = 1
        else:
            holiday_init = 2

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
        holiday_data[index:index + 1, :] = holiday_init

    if holiday_list is None:
        k = 1
    else:
        for j in holiday_list:
            holiday_data[j - 1 * time_slot:j * time_slot, :] = 2
    return day_data, week_data, holiday_data

class TSForecasting(ScenarioAbstract):
    def parse_data(self, data_name, data_path, algorithm):
        data = None
        return data
