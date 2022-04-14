import xml.etree.ElementTree as ET


class XmlHelper(object):
  def __init__(self, tensor_xml_path, udf_xml_path, hyper_parameter_dict_path=""):
    """
    该类用于帮助xml文件的解析、自洽性判定等任务。目前主要用于线上模型服务配置文件的解析、生产
    :param tensor_xml_path: 特征矩阵生成配置文件
    :param udf_xml_path: dispatch服务特征处理UDF配置文件
    :param hyper_parameter_dict_path: UDF对应超参数配置文件
    """
    self.tensor_xml_path = tensor_xml_path
    self.udf_xml_path = udf_xml_path
    self.hyper_parameter_dict_path = hyper_parameter_dict_path

  def _extract_tensor_xml_feature_list(self):
    tensor_tree = ET.parse(self.tensor_xml_path)
    tensor_root = tensor_tree.getroot()
    fields_element = tensor_root.find("features")
    tensor_feature_list = list()
    for field in fields_element.findall("field"):
      if "," in field.find("feature").text:
        tensor_feature_list.extend(field.find("feature").text.split(","))
      else:
        tensor_feature_list.append(field.find("feature").text)
    return tensor_feature_list

  def _extract_field_set_from_tensor_xml_path(self):
    tensor_tree = ET.parse(self.tensor_xml_path)
    tensor_root = tensor_tree.getroot()
    fields_element = tensor_root.find("features")
    tensor_xml_set = set()
    for field in fields_element.findall("field"):
      if field.find("name").text not in tensor_xml_set:
        tensor_xml_set.add(field.find("name").text)
      else:
        raise RuntimeError(f"duplicated field config for {field.find('name').text} in {self.tensor_xml_path}")
    return tensor_xml_set

  def _extract_udf_xml_output_feature_set(self):
    udf_tree = ET.parse(self.udf_xml_path)
    udf_root = udf_tree.getroot()
    features_element = udf_root.find("features")
    udf_feature_set = set()
    udf_duplicated_feature_list = list()
    for feature in features_element.findall("feature"):
      if feature.find("name").text not in udf_feature_set:
        udf_feature_set.add(feature.find("name").text)
      else:
        udf_duplicated_feature_list.append(feature.find("name").text)
    if udf_duplicated_feature_list:
      raise RuntimeError(f"duplicated features in {self.udf_xml_path}: {udf_duplicated_feature_list}")
    return udf_feature_set

  def _load_hyper_parameter_dict(self):
    dic = dict()
    if not self.hyper_parameter_dict_path:
      raise RuntimeError("Please set a hyper_parameter_dict_path first!")
    with open(self.hyper_parameter_dict_path, "r") as f:
      for line in f:
        if line.strip():
          if len(line.strip().split(":")) != 2:
            raise RuntimeError(f"line is not like 'x:y' format: {line}")
          config, value = line.strip().split(":")
          if len(config.split(",")) != 2:
            raise RuntimeError(f"config is not like 'name,operator' format: {config}")
          name, operator = config.split(",")
          if operator not in dic:
            dic[operator] = {name: value}
          else:
            # 配置中可能出现重复，此处暂取位置靠后的配置作为实际使用配置
            # if name in dic[operator]:
            #   raise RuntimeError(f"duplicated config for operator: {operator}, feature: {name}")
            dic[operator][name] = value
      return dic

  def is_tensor_features_supported_by_udf(self):
    """检查tensors_xml中用到的特征在model_name_xml中是否都有生产"""
    tensor_feature_list = self._extract_tensor_xml_feature_list()
    udf_output_feature_set = self._extract_udf_xml_output_feature_set()

    missing_feature_set = set(tensor_feature_list) - udf_output_feature_set
    if missing_feature_set:
      print(f"not supported features like: {missing_feature_set}")
    else:
      print(f"{self.udf_xml_path} supports features for {self.tensor_xml_path}")
    return len(missing_feature_set) == 0

  def remove_useless_feature_in_udf_xml(self):
    """如果udf_xml生产的特征tensor_xml中没有遇到，则去掉这部分特征"""
    tensor_feature_list = self._extract_tensor_xml_feature_list()
    tensor_feature_set = set(tensor_feature_list)
    udf_tree = ET.parse(self.udf_xml_path)
    udf_root = udf_tree.getroot()
    features_element = udf_root.find("features")
    removed_feature_list = list()
    for feature in features_element.findall("feature"):
      if feature.find("name").text not in tensor_feature_set:
        removed_feature_list.append(feature.find("name").text)
        features_element.remove(feature)
    if removed_feature_list:
      print(f"remove {len(removed_feature_list)} useless features: {removed_feature_list}")
      udf_tree.write(self.udf_xml_path)
    else:
      print(f"skip removing: no useless features in {self.udf_xml_path} considering {self.tensor_xml_path}")

  def is_split_embedding_hyper_parameter_supported(self):
    """验证UDF配置正确性：model_name.xml中配置split_embedding归一化特征所需超参数在hypter_parameter_dict文件中均能找到"""
    udf_tree = ET.parse(self.udf_xml_path)
    udf_root = udf_tree.getroot()
    features_element = udf_root.find("features")
    split_embedding_udf_feature_set = set()
    for feature in features_element.findall("feature"):
      if "NormalizeFeature" in feature.find("f").text and "split_embedding" in feature.find("f").text:
        function_text = feature.find("f").text
        split_embedding_feature = function_text[18: function_text.find(",") - 1]
        if split_embedding_feature in split_embedding_udf_feature_set:
          raise RuntimeError(f"duplicated embed split config for feature: {split_embedding_feature}")
        split_embedding_udf_feature_set.add(split_embedding_feature)

    hyper_parameter_dict = self._load_hyper_parameter_dict()
    split_embed_hyper = hyper_parameter_dict["percentile_split"]
    split_embedding_hyper_feature_set = set(split_embed_hyper.keys())

    missing_feature_set = split_embedding_udf_feature_set - split_embedding_hyper_feature_set
    if missing_feature_set:
      print(f"split embedding not supported features number: {len(missing_feature_set)}, "
            f"content: {missing_feature_set}")
    else:
      print(f"{self.hyper_parameter_dict_path} supports split embedding features for {self.udf_xml_path}")
    return len(missing_feature_set) == 0

  def check_field_match(self, model_field_set):
    """查看模型离在线用的field集合差异"""
    tensor_field_set = self._extract_field_set_from_tensor_xml_path()
    print(f"tensor_field_set - model_field_set: {tensor_field_set - model_field_set}")
    print(f"model_field_set - tensor_field_set: {model_field_set - tensor_field_set}")
