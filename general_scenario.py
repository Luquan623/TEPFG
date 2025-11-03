import os
import pandas as pd
import numpy as np

from ts_forecasting_traffic.scenario import ScenarioAbstract

import zipfile

def time_add(data, week_start, interval=5, weekday_only=False, holiday_list=None, day_start=0, hour_of_day=24):
    # day and week
    if weekday_only:
        week_max = 5
    else:
        week_max = 7
    time_slot = hour_of_day * 60 // interval # 计算一天又多少个时间段
    # 创建与 data 形状相同的零矩阵，用于存储处理后的日、周和假期数据。
    day_data = np.zeros_like(data)
    week_data = np.zeros_like(data)
    holiday_data = np.zeros_like(data)

    day_init = day_start  # 表示当前日时间，初始化为0
    week_init = week_start # 表示当前周时间，初始化为1
    holiday_init = 1

    for index in range(day_start//interval, data.shape[0]+day_start//interval): # 遍历数据的所有时间点
        if (index) % time_slot == 0 and index!=0: # 如果当前索引 index 可以整除 time_slot（意味着到达了一天的结束时间），则重置 day_init 为 0，表示新的一天开始。
            day_init = 0
        day_init = day_init + interval # 每次循环都将 day_init 按照 interval 增加。即每隔一个时间间隔，更新一天的时间点。
        if (index) % time_slot == 0 and index !=0: # 同样地，如果索引 index 整除 time_slot，说明新的一天开始，同时 week_init 增加 1，表示一周中的天数变化。
            week_init = week_init + 1
        if week_init > week_max: # 如果当前周天数超过 week_max（即超过一周的天数，比如 5 或 7），则重置 week_init 为 1，表示重新开始新的一周。
            week_init = 1
        # 根据当前 day_init 的值决定是否是假期
        if day_init < 6: # 如果 day_init 小于 6，表示工作日，holiday_init 设置为 1。
            holiday_init = 1
        else: # 否则，设置为 2，表示假期或非工作日。
            holiday_init = 2

        day_data[index:index + 1, :] = day_init
        week_data[index:index + 1, :] = week_init
        holiday_data[index:index + 1, :] = holiday_init

    if holiday_list is None: # 如果 holiday_list 为空，则 k=1，此段代码无实际影响。
        k = 1
    else: # 如果 holiday_list 不为空，则遍历假期列表 holiday_list，并在假期对应的时间段将 holiday_data 设置为 2，表示假期。
        for j in holiday_list :
            holiday_data[j-1 * time_slot:j * time_slot, :] = 2
    return day_data, week_data, holiday_data

class TSForecasting(ScenarioAbstract):
    def parse_data(self, data_name, data_path, algorithm):
        data = None
        return data