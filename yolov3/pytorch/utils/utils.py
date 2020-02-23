def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions
        1. 去除注释项.
        2. 去除两端空格.
        3. 返回一个列表. 其中每一块为一个词典，开头为type，后面是等号左右两侧对应key-value
            注意卷积层的类中，有'batch_normalize'项
    """
    module_defs = []

    # “r+”读写操作
    # f = open(path, "r+")
    f = open(path, 'r')

    # read([size]) 不指明size时，读取整个文件内容，注意内存问题
    # contents = f.readlines()
    contents = f.read().split('\n')
    f.close()

    # get rid of whitespaces
    contents = [x for x in contents if x and not x.startswith('#')]
    contents = [x.strip() for x in contents]

    for line in contents:
        # do this before loop
        # line = line.strip()
        # if line.startswith("#") or line == "":
        #     continue
        if line.startswith('['):
            record = {}

            # get rid of all possible whitespaces
            record['type'] = line[1:-1].rstrip()
            
            # only convolutional cares about batch_normalize
            if record['type'] == 'convolutional':
                record['batch_normalize'] = 0
            module_defs.append(record)
        else:
            # know split return two value
            # line_part = line.split("=")
            # module_defs[-1][line_part[0]] = line_part[1]
            key, value = line.split("=")
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs
