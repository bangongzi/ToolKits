import os.path
from datetime import datetime
from datetime import timedelta

def date_alter(start_date, day_delta, date_format='%Y%m%d'):
    time = datetime.strptime(start_date, date_format)
    time = time + timedelta(days=day_delta)
    return time.strftime(date_format)

def generate_data_path(data_path_prefix, start_date, day_num):
    """生成指定时间范围的数据地址"""
    data_path_list = list()
    for i in range(day_num):
        date = date_alter(start_date, i)
        path = os.path.join(data_path_prefix, date)
        data_path_list.append(path)
    return data_path_list


def date_range(start_date, end_date, date_format='%Y%m%d'):
  """生成指定起始日期和结束日期范围内的日期列表"""
  start_date = datetime.strptime(start_date, date_format)
  end_date = datetime.strptime(end_date, date_format)
  if end_date < start_date:
    raise ValueError(f"end_date({end_date}) should not be smaller than start_date({start_date})")
  ret_list = []
  for i in range((end_date - start_date).days + 1):
    inner_date = start_date + timedelta(days=i)
    ret_list.append(inner_date.strftime(date_format))
  return ret_list

def partition_check(table_name, start_date, end_date, source_dir='data/partition_check/'):
  """检查指定日期范围内的数据表每天的数据是否正常，检查内容包括date字符是否存在，以及date下数据量大于0"""
  number_list, unit_list, date_list = [], [], []
  with open(os.path.join(source_dir, table_name), 'r') as f:
    for line in f.readlines():
      number, unit, date = line.split(' ')[0], line.split(' ')[1], line.strip()[-8:]
      number = float(number)
      number_list.append(number)
      unit_list.append(unit)
      date_list.append(date)

  def check_date(date_list, start_date, end_date, table_name):
    target_date_list = date_range(start_date, end_date)
    missing_date_list = list()
    for date in target_date_list:
      if date not in date_list:
        missing_date_list.append(date)
    if len(missing_date_list) > 0:
      raise ValueError(f"{table_name}: Missing {len(missing_date_list)} dates {missing_date_list}")

  check_date(date_list, start_date, end_date, table_name)

  # todo 可进一步加入对每天数据量变化门限值的check
  if min(number_list) <= 0:
    raise ValueError(f"{table_name}: have empty partitions")

  if len(set(unit_list)) > 1:
    print(f"Exception! {table_name}: have more than one units: {set(unit_list)}")
    # raise ValueError(f"{table_name}: have more than one units: {set(unit_list)}")

  print(
    f"{table_name}: every dt between [{start_date}, {end_date}] having data, avg {round(sum(number_list) / len(number_list), 2)}{unit_list[0]}")

if __name__ == '__main__':
  """source file like (hdfs dfs -du -h your_hdfs_path):
11.3 G  viewfs://hadoop-xxx/some_table/dt=20211001
11.0 G  viewfs://hadoop-xxx/some_table/dt=20211002
12.0 G  viewfs://hadoop-xxx/some_table/dt=20211003
12.0 G  viewfs://hadoop-xxx/some_table/dt=20211005"""
  partition_check("some_table_name", "20211001", "20211015")
  
  generate_data_path("tf_sample_path", "20221001", 10)
