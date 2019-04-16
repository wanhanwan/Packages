import pandas as pd


def read_csv(file_path, **kwargs):
    """读取一个csv文件

    为pandas.read_csv()加一个上下文管理器，
    因为读取csv文件的时候不支持中文路径.

    Parameters:
    ===========
    file_path: str or Path object
        csv文件路径
    **kwargs:
        pandas.read_csv()函数中的其他参数
    """
    try:
        with open(file_path) as f:
            data = pd.read_csv(f, **kwargs)
    except Exception as e:
        print(e)
        raise ValueError("读取文件失败！")
    return data


def write_excel(df_dict, file_path, **kwargs):
    with pd.ExcelWriter(file_path, **kwargs) as f:
        for df in df_dict:
            df_dict[df].to_excel(f, sheet_name=df)


def read_wind_excel(file_name, usecols=None):
    """读取Wind插件提取的Excel数据"""
    data = pd.read_excel(file_name, header=0, index_col=0,
                         parse_dates=True, skiprows=[1,2,3])
    data.index.name = 'date'
    if usecols is not None:
        return data[list(usecols)].rename(columns=usecols)
    return data
