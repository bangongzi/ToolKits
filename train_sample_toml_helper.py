import re
import toml

"""该文件主要用于数据辅助表、数据计算UDF都具备的情况下，快速生成样本生成配置文件，并配置到模型中
   具体包括：
   特征拼接、特征计算、TFRecords配置
   模型侧reader, identity配置"""

# 旧样本流程下配置生成工具
def extract_name_type_from_schema(schema_str):
  """用于从hive表的shema配置中提取列名和列类型
     参考schema配置：
     user_id string COMMENT '用户Id',
     user_price double COMMWNT '用户价格偏好'"""
  item_list = [item.strip() for item in schema_str.split(',')]
  feature_list = list()
  for item in item_list:
    lst = [word.strip() for word in item.split(' ') if word.strip()]
    feature_list.append((lst[0], lst[1]))
  return feature_list

def generate_select_columns(feature_list):
  """将schema中提取的列信息转化为toml配置特征join时列的提取配置"""
  feature_list = [(item[0], "long" if item[1] == "bigint" else item[1]) for item in feature_list]
  valid_type = ["int", "long", "double", "string"]
  column_list = list()
  for name, field_type in feature_list:
    if field_type not in valid_type:
      raise TypeError("not valid type")
    column_list.append("    \"{}:{}\"".format(name, field_type))
  print(",\n".join(column_list))

def extract_type_name_info_from_columns(column_list):
  """从generate_select_columns()产生的提取配置中汇总各个数据类型下的特征字段
     支持'col1 as col2:double'这种有重命名的字段提取"""
  type_name_dict = dict()
  for column in column_list:
    try:
      name, field_type = column.split(":")
      name = name.split(" ")[-1]
      if field_type not in type_name_dict:
        type_name_dict[field_type] = [name, ]
      else:
        type_name_dict[field_type].append(name)
    except ValueError:
      print("Fail to parse column: {}".format(column))
      raise ValueError
  return type_name_dict


def generate_equal_op_from_dict(type_name_dict):
  """从上述提取到的字段字典产生eaual op配置"""
  equal_format = """[[FirstOp]]
    name = "EqualOperator"
    type = "{}"
    input_columns = [
{}
    ]"""
  for filed_type, name_list in type_name_dict.items():
    name_str = ",\n".join(["        \"{}\"".format(item) for item in name_list])
    print(equal_format.format(filed_type, name_str) + "\n")


def generate_split_embedding_op_from_two_dict(all_type_name_dict, exclude_type_name_dict):
  """从上述提取到的字段字典产生splitEmbedding op配置"""
  norm_format = """[[SecondOp]]
    name = "SplitEmbeddingNormOperator"
    type = "{}"
    input_columns = [
{}
    ]"""
  for field_type in ["long", "int", "double"]:
    if field_type in all_type_name_dict:
      name_list = all_type_name_dict[field_type]
      if field_type in exclude_type_name_dict:
          name_list = [item for item in name_list if item not in exclude_type_name_dict[field_type]]
      name_str = ",\n".join(["        \"{}\"".format(item) for item in name_list])
      print(norm_format.format(field_type, name_str))


def generate_tf_config_from_dict(type_name_dict):
  """从上述提取到的字段字典中产生TFRecords配置"""
  tf_format = """[[TFRecord.Field]]
    name="{name}"
    type="{type}"
    columns=["{name}"]"""

  for filed_type, name_list in type_name_dict.items():
    for name in name_list:
      print(tf_format.format(name=name, type=filed_type))

def extract_feature_dict_from_toml_tfrecord(toml_path):
  """从样本生成的toml文件中获取生成数据列的类型以及特征名"""
  data = toml.load(toml_path)
  type_name_dict = dict()
  for item in data['TFRecord']['Field']:
    if item['type'] not in type_name_dict:
      type_name_dict[item['type']] = [item['name'], ]
    else:
      type_name_dict[item['type']].append(item['name'])
  return type_name_dict

def generate_reader_lines_from_feature_dict(feature_dict):
  """由上面的数据dict生成reader中的数据列读取文件
     此处需要注意，生成的TFRecords文件中类型为long，才能用tf.int64这个类型读取；
     若TFRecord中类型为int，或者在上游输出的值是double，被强制声明为long，用tf.int64这个类型读取时，都会报错xx_feature (tf.int64) not found"""
  field_type_alias_map = {"int": "tf.int64", "long": "tf.int64", "double": "tf.float32"}
  for name1, name2 in field_type_alias_map.items():
    if name1 in feature_dict:
      if name2 not in feature_dict:
        feature_dict[name2] = feature_dict[name1]
      else:
        feature_dict[name2].extend(feature_dict[name1])
      del feature_dict[name1]

  print(feature_dict)
  interested_field_type = ["tf.int64", "tf.float32"]
  for field_type in interested_field_type:
    if field_type in feature_dict:
      featue_list = feature_dict[field_type]
      for featue in featue_list:
        print("            '{}': tf.FixedLenFeature([1], {}),".format(featue, field_type))

def genearte_identity_from_feature_dict(feature_dict):
  """由上面的数据dict生成"""
  for field_type, name_list in feature_dict.items():
    if field_type in ('int', 'long', 'double'):
      for feature_name in name_list:
        if feature_name not in {"age", "city_code", "nation_code"}:
          if 'user' in feature_name:
            print('    user_matrix_dict[\"{name}\"] = tf.identity(data_batch[\"{name}\"], \"{name}\")'.format(name=feature_name))
          elif 'poi' in feature_name:
            print('    poi_matrix_dict[\"{name}\"] = tf.identity(data_batch[\"{name}\"], \"{name}\")'.format(name=feature_name))
          else:
            print('    x_matrix_dict[\"{name}\"] = tf.identity(data_batch[\"{name}\"], \"{name}\")'.format(name=feature_name))

# 以下为旧样本生成流程debug工具
def extract_input_columns_from_tf_config(config_file_path):
  tf_column_list = list()
  with open(config_file_path, "r") as f:
    line = f.readline()
    while line:
      if "columns" in line:
        try:
          column_txt = re.findall(r"\[(.+?)\]", line)[0]
          columns_list = [item.strip().replace("\"", "") for item in column_txt.split(',')]
          tf_column_list.extend(columns_list)
        except:
          print(line)
      line = f.readline()
  return tf_column_list

def list_item_compare(list1, list2, name1="list1", name2="list2"):
  print("{} - {}: {}".format(name1, name2, set(list1) - set(list2)))
  print("{} - {}: {}".format(name2, name2, set(list2) - set(list1)))

if __name__ == '__main__':
  column_list = [
    "poi_x_hour_pay_cnt_180d:long",
    "poi_x_hour_pay_uv_180d:long",
    "poi_x_hour_gmv_180d:double",
    "poi_x_hour_pay_cnt_60d:long",
    "poi_x_hour_pay_uv_60d:long",
    "poi_x_hour_gmv_60d:double",
    "poi_x_mealtime_ck_cnt_180d:long"
  ]
  type_name_dict = extract_type_name_info_from_columns(column_list)
  generate_equal_op_from_dict(type_name_dict)
  norm_exclude_dic = dict()
  generate_split_embedding_op_from_two_dict(type_name_dict, norm_exclude_dic)
  # generate_equal_op_from_dict(norm_exclude_dic)
  generate_tf_config_from_dict(type_name_dict)

  schema_txt = """poi_id string,
  mealtime int,
  poi_x_mealtime_pv_7d bigint,
  poi_x_mealtime_pv_30d bigint,
  poi_x_mealtime_pv_60d bigint,
  poi_x_mealtime_pv_180d bigint,
  poi_x_mealtime_ck_cnt_7d bigint,
  poi_x_mealtime_ck_cnt_30d bigint,
  poi_x_mealtime_ck_cnt_60d bigint,
  poi_x_mealtime_ck_cnt_180d bigint"""
