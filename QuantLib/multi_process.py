import numpy as np
from multiprocessing.pool import Pool


def run_func(target_func, func_args, split_args, core_nums=2):
    """多线程运算

    Parameters
    -----------
    target_func: func
        待运行的函数。需要分配到不同进程的参数必须放在该函数参数列表的最前面，即：
        target_func(split_args, func_args)
    func_args: dict
        被传入到运行函数中
    split_args: 2-dimensioned array of N * K
        参数列表会平均分配到不同的进程中去。N代表参数的个数，K代表每个参数的长度
    core_nums: int
        创建进程的数量
    """
    s_args = np.array_split(split_args, core_nums, axis=1)
    p = Pool(core_nums)
    for i in range(core_nums):
        print("create process %s" % i)
        p.apply_async(target_func, args=tuple(s_args[i]), kwds=func_args)
    p.close()
    p.join()
    print("calculation has finished!")
    