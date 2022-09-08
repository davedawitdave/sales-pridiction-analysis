import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api as dvc

import io 
import sys
sys.path.append('../')
from logger import Logger


class Summery:
       def __init__(self) -> None:
            """Initilize class."""
            try:
                pass
                self.logger = Logger("utility.log").get_app_logger()
                self.logger.info(
                    'Successfully initialized Object')
            except Exception:
                self.logger.exception(
                    'Failed to initialized Object')
                sys.exit(1)
       
       def read_from_file(self,path,low_memory=True):
            """
                Load data from a csv file
            """
            try:
                df = pd.read_csv(path)
                self.logger.info(f"successsfuly read {path}")
                return df
            except FileNotFoundError:
                self.logger.error(f"failed to read {path}; file not found")

                print("File not found.")
       def read_from_dvc(self,path,repo,rev,low_memory=True):
            
            """
                Load data from a dvc storage
            """
            try:
                data = dvc.read(path=path,repo=repo, rev=rev)
                df = pd.read_csv(io.StringIO(data),low_memory=low_memory)
                self.logger.info(f"successsfuly read {path} from dvc")

                return df
            except Exception as e:
                self.logger.error(f"failed to read {path}; {e}")

                print("Something went wrong!",e)