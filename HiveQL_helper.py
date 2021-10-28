def shema_generator(name_str, type_str, comment_str, len1=30, len2=15, len3=12):
  """该函数主要目的是生成表格的Schema文件"""
  name_list = name_str.split(',')
  type_list = type_str.split(',')
  comment_list = comment_str.split(',')
  for i in range(len(name_list)):
    name, type, comment = name_list[i], type_list[i], comment_list[i]
    name_part = '  ' + name + ''.join([' ' for i in range(len1 - len(name))])
    type_part = type + ''.join([' ' for i in range(len2 - len(type))])
    COMMENT_STR = 'COMMENT'
    comment_part = "{}{}'{}',".format(COMMENT_STR, ''.join([' ' for i in range(len3 - len(COMMENT_STR))]), comment)
    print(name_part + type_part + comment_part)
    # print("  {}\t{}\tCOMMENT\t'{}'".format(name, type, comment))


def test_shema_generator():
  name_str = 'ctr,uv,user_id'
  type_str = 'double,bigint,string'
  comment_str = '点击率,浏览人数,用户Id'
  shema_generator(name_str, type_str, comment_str)
  

if __name__ == '__main__':
  test_shema_generator
