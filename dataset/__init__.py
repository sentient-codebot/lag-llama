import enum


from dataset.utils import NAME_SEASONS, TimeSeriesDataset, PIT, standard_normal_cdf, standard_normal_icdf
from dataset.heat_pump import WPuQ
from dataset.wpuq_trafo import WPuQTrafo
from dataset.wpuq_pv import WPuQPV
from dataset.lcl_electricity import LCLElectricityProfile
from dataset.cossmic import CoSSMic

from dataset.wpuq_pv import DIRECTION_CODE as WPUQ_PV_DIRECTION_CODE

# class DatasetType(enum.Enum):
#     HEAT_PUMP = enum.auto()
#     LCL_ELECTRICITY = enum.auto()

all_dataset = {
    'wpuq',
    'lcl_electricity',
    'cossmic',
    'wpuq_trafo',
    'wpuq_pv'
}

