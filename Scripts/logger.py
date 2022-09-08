from distutils.debug import DEBUG
import unittest
import numpy as np
import pandas as pd
import sys, os
 
# importing scripts
sys.path.insert(1, '..')
sys.path.append("..")
sys.path.append(".")

from Scripts import data_summery
from Scripts import data_cleaner
DV = data_summery.Data_summery("logs/test.log")
DC = data_cleaner.Data_cleaner("logs/test.log")

if __name__ == '__main__':
	unittest.main()
    