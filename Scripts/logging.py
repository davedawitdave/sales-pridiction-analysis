from distutils.debug import DEBUG
import logging 

LOG_FORMAT="%(levelname)s %(asctime)s - %(message)s"
logging. basicConfig(filename=C:/Users/User/anaconda3/envs/10_20/Sals-pridiction_analysis/sales-pridiction-analysis/log/data_exploration.log, level=logging.DEBUG, fromat=LOG_FORMAT)
logger = logging.getLogger()

#test the logger
logger.info("commit happen at")

print(logger.level)

