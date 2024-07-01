import numpy as np


ed_pressure = 80/7.50062   # kPa
es_pressure = 130.0/7.50062 # kPa

mbp = (es_pressure + 2*ed_pressure)/3
print(mbp*7.50062, mbp**2/ed_pressure*7.50062)