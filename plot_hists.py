import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np

plt.hist(Table.read('outputcat_daopy',format='ascii')['flux'])
plt.hist(Table.read('outputcat_dao',format='ascii')['flux'])
plt.show()