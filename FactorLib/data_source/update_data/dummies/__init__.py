import pandas as pd
from pathlib import Path


def _read_dummy_class():
    file_path = Path(__file__).parent.parent.parent / 'resource' / 'user_dummy.xlsx'
    mapping = pd.ExcelFile()
    return mapping


