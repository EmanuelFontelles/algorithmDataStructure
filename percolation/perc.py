#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Eric Stansifer
# Date: 2017 March 19
# License: MIT License
# Hosted on https://github.com/estansifer/percolation
# Tested on Python 3.5.2. Requires numpy. Seems to also work on Python 2.7.

import numpy as np
import numpy.random as npr
import sys

#
# Useful functions in this module:
#
# biggest_clusters(d, Ls, ps, numclusters)
#   d: integer (>= 1), the dimension
#   Ls: integer or list of integers (>= 2), the grid size
#   ps: float or list of floats (>= 0, <= 1), the occupancy probability
#   numclusters: integer, how many of the biggest clusters to return
#   numclusters is optional and 10 by default
#   At most one of Ls or ps can have more than one number
#   returns a list of triples (L, p, [size]) where [size] is a list
#   in decreasing order of cluster sizes for the indicates L and p.
#   (The results will be returned in order of increasing L or p.)
#
# percolation2d(keyword, L, p)
#   keyword: either "ones", "masses", "biggest", or "whichcluster"
#   L: integer (>= 2), the grid size
#   p: float (>= 0, <= 1), the occupancy probability
#   returns LxL grid of integers. Meaning of the grid depends on choice
#   of the keyword. See usage for more information.
#

suppress_warning = True

usage=(
"""A program for finding sizes of clusters in a percolation process.
Usage:
    python perc.py d L p [numclusters]
or
    python perc.py d L1,L2,...,Ln p [numclusters]
or
    python perc.py d L p1,p2,...,pn [numclusters]
or
    python perc.py keyword L p [filename]

The first argument should be an integer d >= 1 indicating the dimesion of the
percolation process.
The second argument should be either an integer L >= 2 or a comma-separated list
of such integers, indicating the size of the grid used for the percolation
process.
The third argument should be either a floating point number 0 <= p <= 1 or a
comma-separated list of such numbers, indicating the probability of each site
being filled in the percolation process.
The fourth argument is optional, and should be an integer >= 1 indicating how
many of the cluster sizes to print out in the output. By default it is 10.

Either the grid sizes L or the probabilities p can be given as a list of
multiple values, but not both.

For each d, L, p, a percolation process is performed on a d-dimensional grid
where each site is filled with probability p and clusters of contiguous
filled sites are found. The sizes of the largest clusters in the grid are
printed. The grid wraps around the ends, i.e., it is a torus.

If multiple values for L or p are given, the percolation process is only
performed for the largest value of L or p and the other values are found
along the way. This makes it possible to efficiently find large clusters for
many different values of L and p simultaneously.

If instead of a dimension, a keyword is given, then a percolation process is
run on a 2D grid of size L x L with probability p. The percolation grid is
printed in the given file (or printed to output if no filename is given), with
a zero in every unoccupied grid square and a positive integer in every
occupied grid square.
    For keyword "ones": 1 is printed in every occupied square.
    For keyword "masses": in every occupied grid square, the mass of the
        cluster it is in is printed.
    For keyword "biggest": 2 is printed in every square occupied by the
        largest cluster, and 1 is printed in every square occupied by another
        cluster. If there is more than one largest cluster, one is chosen
        arbitrarily.
    For keyword "whichcluster": the same positive integer is printed in
        two grid squares if and only if they are part of the same cluster.

For example, using the biggest keyword you could find the mass of the largest
cluster by computing
    (sum(xs ^ 2) - sum(xs)) / 2
where 'xs' is the grid that was output. Using the whichcluster keyword,
the maximum of the result is equal to the number of clusters.

On my machine, the running time of this program is approximately
    2 * d * (max L)^d * (max p) * 10^-6
seconds.""")

def prettyprint(data):
    nrow = len(data)
    ncol = max([len(row) for row in data])
    colsize = [0] * ncol
    for i in range(nrow):
        for j in range(ncol):
            if len(data[i][j]) > colsize[j]:
                colsize[j] = len(data[i][j])

    result = []
    for row in data:
        cur = []
        for j, value in enumerate(row):
            cur.append(' ' * (colsize[j] - len(value)) + value)
        result.append('  '.join(cur) + '\n')
    return ''.join(result)

# Given:
#   a positive integer 'd', the dimension
#   an integer 'L' >= 2, the length scale
#   a strictly increasing list 'ps' of real numbers 0 < 'p' < 1
# Returns:
#   a list, for each element of 'ps', of
#       a triple (L, p, s) where
#           'L' is the given size L,
#           'p' is the element of 'ps', and
#           's' is a list in decreasing order of cluster sizes
#
# This function performs a single test of a percolation process
# in d dimensions of an L x L x ... x L size grid. Each cell in the
# grid is randomly occupied with probability p or unoccupied with
# probability 1 - p. All clusters of touching occupied cells are
# found, and the sizes of the largest clusters are returned in a list.
#
# The 'numclusters' variable changes how many of the largest clusters
# will be returned.
#
# The grid wraps around the edges, i.e., it is a torus.
#
def biggest_clusters_ps(d, L, ps, numclusters = 10, interrupt = False):
    V = L ** d
    if (d * V * max(ps) > 1e7) and not suppress_warning:
        sys.stderr.write("A very large number {} of cells was chosen in the domain, this may take a long time.".format(V))

    Vd = V * d

    cells = np.zeros((V, 2 * d), dtype = int)
    for i in range(d):
        cells[:, i] = np.arange(V, dtype = int) * d + i
        cells[:, d + i] = cells[:, i] + (L ** i) * d
    for i in range(d):
        idx = np.arange(L ** (d - 1), dtype = int)
        Li = L ** i
        idx = (np.mod(idx, Li) + (Li * (L * np.floor_divide(idx, Li) + (L-1))))
        cells[idx, d + i] -= Li * L * d

    q = npr.random((V,))
    order = np.argsort(q)
    cells = cells[order]

    ps = sorted(ps)
    indices = np.searchsorted(q[order], ps)

    uf = np.arange(Vd, dtype = int) # union-find data structure
    mass = np.zeros((Vd,), dtype = int)

    # def root(i):
        # while uf[i] != i:
            # uf[i] = uf[uf[i]]
            # i = uf[i]
        # return i

    results = []
    for k in range(len(ps)):
        if k == 0:
            start = 0
        else:
            start = indices[k - 1]
        for c in cells[start:indices[k]]:

            # c0 = root(c[0])
            c0 = c[0]
            while uf[c0] != c0:
                uf[c0] = uf[uf[c0]]
                c0 = uf[c0]

            mass[c0] += 1
            for ci in c[1:]:
                # ci = root(ci)
                while uf[ci] != ci:
                    uf[ci] = uf[uf[ci]]
                    ci = uf[ci]

                if c0 != ci:
                    # Join
                    if mass[c0] < mass[ci]:
                        mass[ci] += mass[c0]
                        uf[c0] = ci
                        c0 = ci
                    else:
                        mass[c0] += mass[ci]
                        uf[ci] = c0

        if interrupt:
            return (cells[0:indices[0], 0], uf, mass)

        # Find the most massive clusters
        masses = list(mass[np.arange(Vd, dtype = int) == uf])
        masses.sort(reverse = True)
        results.append([L, ps[k], masses[:numclusters]])

    return results

#
# Instead of taking a list of values for 'p', take a single value
# for 'p' and take a list of values for 'L'.
#
def biggest_clusters_Ls(d, Ls, p, numclusters = 10):
    Lmax = Ls[-1]

    V = Lmax ** d
    Vd = V * d
    if (d * V * p > 1e7) and not suppress_warning:
        sys.stderr.write("A very large number {} of cells was chosen in the domain, this may take a long time.".format(V))

    cells = np.zeros((V, 2 * d), dtype = int)
    for i in range(d):
        cells[:, i] = np.arange(V, dtype = int) * d + i
        cells[:, d + i] = cells[:, i] + (Lmax ** i) * d
    for i in range(d):
        idx = np.arange(Lmax ** (d - 1), dtype = int)
        Li = Lmax ** i
        idx = (np.mod(idx, Li) + (Li * (Lmax * np.floor_divide(idx, Li) + (Lmax-1))))
        cells[idx, d + i] -= Li * Lmax * d

    # Compute l_infinity distance from origin
    dist = np.zeros((V,), dtype = int)
    cur = np.arange(V, dtype = int)
    for i in range(d):
        dist = np.maximum(dist, np.mod(cur, Lmax))
        cur = np.floor_divide(cur, Lmax)

    order = np.argsort(dist)
    cells = cells[order]

    indices = np.flatnonzero(npr.random((V,)) < p)

    uf = np.arange(Vd, dtype = int) # union-find data structure
    mass = np.zeros((Vd,), dtype = int)

    results = []
    for k in range(len(Ls)):
        L = Ls[k]
        if k == 0:
            start = 0
        else:
            start = Ls[k - 1] ** d

        for c in cells[indices[(indices >= start) & (indices < (L ** d))]]:
            c0 = c[0]
            while uf[c0] != c0:
                uf[c0] = uf[uf[c0]]
                c0 = uf[c0]

            mass[c0] += 1
            for ci in c[1:]:
                while uf[ci] != ci:
                    uf[ci] = uf[uf[ci]]
                    ci = uf[ci]

                if c0 != ci:
                    # Join
                    if mass[c0] < mass[ci]:
                        mass[ci] += mass[c0]
                        uf[c0] = ci
                        c0 = ci
                    else:
                        mass[c0] += mass[ci]
                        uf[ci] = c0

        # Connect boundaries to make a torus
        uf2 = np.copy(uf)
        mass2 = np.copy(mass)
        if L < Lmax:
            for i in range(d):
                # We cheat a little, this is fine for d <= 3 but it costs a very little bit of
                # extra time for d > 3
                if i > 0:
                    low = np.arange(Lmax ** (i - 1) * L)
                    low = low[np.floor_divide(low, Lmax) < L]
                else:
                    low = [0]
                if d - i - 1 > 0:
                    high = np.arange(Lmax ** (d - i - 2) * L)
                    high = high[np.floor_divide(high, Lmax) < L]
                else:
                    high = [0]

                Li = Lmax ** i
                for a in low:
                    for b in high:
                        c1 = (a + Li * Lmax * b) * d + i
                        c2 = c1 + Li * L * d
                        # Join c1 and c2
                        while uf2[c1] != c1:
                            uf2[c1] = uf2[uf2[c1]]
                            c1 = uf2[c1]
                        while uf2[c2] != c2:
                            uf2[c2] = uf2[uf2[c2]]
                            c2 = uf2[c2]
                        if c1 != c2:
                            mass2[c1] += mass2[c2]
                            uf2[c2] = c1

        # Find the most massive clusters
        masses = list(mass2[np.arange(Vd, dtype = int) == uf2])
        masses.sort(reverse = True)
        results.append([L, p, masses[:numclusters]])

    return results

def biggest_clusters(d, Ls, ps, numclusters = 10):
    assert(type(d) is int)
    assert(d >= 1)
    assert(type(numclusters) is int)
    assert(numclusters >= 1)

    Ls = np.array(Ls, copy = True, dtype = int).flatten()
    ps = np.array(ps, copy = True, dtype = float).flatten()
    Ls.sort()
    ps.sort()

    for L in Ls:
        assert(L >= 2)
    for p in ps:
        assert((0 <= p) and (p <= 1))

    if len(ps) == 1:
        result = biggest_clusters_Ls(d, Ls, ps[0], numclusters)
    elif len(Ls) == 1:
        result = biggest_clusters_ps(d, Ls[0], ps, numclusters)
    else:
        raise ValueError("Need either Ls or ps to be a singleton, but Ls = {}, ps = {}".format(Ls, ps))

    for r in result:
        if len(r[2]) < numclusters:
            r[2] = list(r[2]) + [0] * (numclusters - len(r[2]))

    return result

def display_biggest_clusters(d, Ls, ps, numclusters = 10, out = sys.stdout):
    result = biggest_clusters(d, Ls, ps, numclusters)

    table = []
    table.append(['L', 'p'] + ['size' + str(i) for i in range(numclusters)])
    for r in result:
        row = [str(size) for size in r[2]]
        row.extend(['0'] * (numclusters - len(row)))
        table.append([str(r[0]), str(r[1])] + row)

    out.write(prettyprint(table))

def percolation2d(keyword, L, p):
    cells, uf, mass = biggest_clusters_ps(2, L, [p], interrupt = True)

    def root(c):
        while uf[c] != c:
            uf[c] = uf[uf[c]]
            c = uf[c]
        return c

    grid = np.zeros((L, L), dtype = int)

    for c in cells:
        i = ((c // 2) % L)
        j = ((c // 2) // L)

        grid[i, j] = root(c) + 1

    if keyword == 'masses':
        for i in range(L):
            for j in range(L):
                if grid[i, j] > 0:
                    grid[i, j] = mass[grid[i, j] - 1]
        return grid
    elif keyword == 'ones':
        grid[grid > 0] = 1
        return grid
    elif keyword == 'whichcluster':
        _, g2 = np.unique(grid, return_inverse = True)
        return g2.reshape(grid.shape)
    elif keyword == 'biggest':
        maxmass = 0
        choice = None
        for i in range(L):
            for j in range(L):
                if grid[i, j] > 0:
                    r = grid[i, j] - 1
                    if mass[r] > maxmass:
                        maxmass = mass[r]
                        choice = r + 1
        for i in range(L):
            for j in range(L):
                if grid[i, j] > 0:
                    if grid[i, j] == choice:
                        grid[i, j] = 2
                    else:
                        grid[i, j] = 1

        return grid
    else:
        raise ValueError('Unrecognized keyword "{}"'.format(keyword))

def display_percolation2d(keyword, L, p, out = sys.stdout):
    grid = percolation2d(keyword, L, p)

    table = [[str(grid[i, j]) for i in range(L)] for j in range(L)]

    out.write(prettyprint(table))

def main():
    args = sys.argv
    # Arguments:
    #   1:  d
    #   2:  Ls
    #   3:  ps
    #   4 (optional): numclusters
    # or
    #   1:  keyword
    #   2:  L
    #   3:  p
    #   4 (optional):  filename
    if 4 <= len(args) <= 5:
        try:
            int(args[1])
        except:
            keyword = args[1].lower()
            L = int(args[2])
            p = float(args[3])

            if len(args) >= 5:
                filename = args[4]
                with open(filename, 'w') as f:
                    display_percolation2d(keyword, L, p, f)
            else:
                display_percolation2d(keyword, L, p)
        else:
            d = int(args[1])
            Ls = [int(L) for L in args[2].split(',')]
            ps = [float(p) for p in args[3].split(',')]

            if len(Ls) > 1 and len(ps) > 1:
                print('There must be either only one grid size or only one probability given. '
                    'Try "python perc.py" for usage information.')
                sys.exit(0)

            if len(args) >= 5:
                numclusters = int(args[4])
                display_biggest_clusters(d, Ls, ps, numclusters)
            else:
                display_biggest_clusters(d, Ls, ps)
    else:
        print(usage)

if __name__ == "__main__":
    main()
