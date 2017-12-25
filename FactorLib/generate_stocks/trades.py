import os
from FactorLib.generate_stocks.templates import FactorTradesListGenerator

cwd = os.getcwd()
os.chdir(os.path.abspath(os.path.dirname(__file__)+'/..'))


def update(start, end):
    g = FactorTradesListGenerator()
    g.update(start, end)
    os.chdir(cwd)

if __name__ == '__main__':
    update('20070101', '20171130')