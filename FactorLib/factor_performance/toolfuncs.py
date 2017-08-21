# coding=utf-8
'''
some tool funcs for performanceanalyze
'''
import pandas as pd
import warnings


def set_datetime_index(series):
    '''
    replace index of series with datetime format
    '''
    series.index = pd.DatetimeIndex(series.index)
    return series


def get_benchmark_return(benchmark_returns, benchmark_name):
    """ get specific benchmark returns from benchmark return dict

    PARAMETERS
    ----------
    benchmark_returns: ditc
    all classes of benchmark returns.
    Dict key is the ID of benchmark(000300 for example)

    benchmark_name:str
    the ID of the benchmark
    """

    if not benchmark_returns:
        get_warning("BenchmarkRet is empty! It will be filled by zero")
        benchmarkRet = 0
    else:
        try:
            benchmarkRet = benchmark_returns[benchmark_name]
        except:
            raise KeyError("No benchmark called %s !" % benchmark_name)

    return benchmarkRet


def get_warning(msg):
    """generate a warning
    """

    warnings.warn(msg)

    return 1

# 默认的对冲标的
DEFAULT_BENCHMARK = '000905'
