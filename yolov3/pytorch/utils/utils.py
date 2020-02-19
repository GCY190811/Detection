def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions
        1. 去除注释项.
        2. 去除两端空格.
        3. 返回一个列表. 其中每一块为一个词典，开头为type，后面是等号左右两侧对应key-value
            注意卷积层的类中，有'batch_normaliz'项
    """
    module_defs = []

    pass

    return module_defs
