# import numpy as np
import pprint

import perc

print("Biggest clusters example 1:")
clusters = perc.biggest_clusters(2, 100, 0.5)
pprint.pprint(clusters)
# 'clusters' is a list of triples;
#   the first value is L,
#   the second value is p,
#   the third value is a list of the biggest cluster sizes for that L and p


print("Biggest clusters example 2:")
clusters = perc.biggest_clusters(2, [30, 40, 50, 100], 0.5)
pprint.pprint(clusters)


print("Biggest clusters example 3:")
clusters = perc.biggest_clusters(2, 100, [0.3, 0.4, 0.5, 0.6, 0.7])
pprint.pprint(clusters)


print("percolation2d example:")
grid = perc.percolation2d('ones', 10, 0.5)
pprint.pprint(grid)
# 'grid' is a 10x10 numpy array
