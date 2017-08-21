# -*- coding: utf-8 -*-

from rqalpha import run_file
from datetime import datetime
import os
import sys, getopt

start = "2007-01-01"
end = datetime.now().strftime("%Y-%m-%d")
stocklist_path = ""

opts, args = getopt.getopt(sys.argv[1:], "s:e:f:")
for op, value in opts:
  if op == "-s":
    start = value
  elif op == "-e":
    end = value
  elif op == "-f":
    stocklist_path = value

config = {
  "base": {
    "start_date": start,
    "end_date": end,
    "accounts": {
      "stock": 100000000
    }
  },
  "extra": {
    "log_level": "verbose",
  },
  "mod": {
    "sys_analyser": {
      "enabled": True,
      "plot": False,
      "output_file": "BTresult.pkl",
      "priority": 101
    },
    "run_csv_signals": {
      "enabled": True,
      "stocklist_path": stocklist_path
    },
    "portfolio_persist": {
      "enabled": True
    }
  }
}

strategy_file_path = "./strategy.py"

os.chdir(os.path.dirname(__file__))
run_file(strategy_file_path, config)
