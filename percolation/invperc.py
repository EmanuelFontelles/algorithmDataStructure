#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Author: Eric Stansifer
# Date: 2017 May 4
# License: MIT License
# Hosted on https://github.com/estansifer/percolation
# Tested on Python 3.5.2. Requires numpy.

import numpy as np
import numpy.random as npr
import sys
import heapq

push = heapq.heappush
pop = heapq.heappop

def invperc(L):
    L = int(L)

    adj = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    visited = np.zeros((L + 1, L, L), dtype = bool)

    rs = npr.random((6 * (L + 1) * L * L,))

    i = 0
    q = []
    for y in range(L):
        for z in range(L):
            q.append((0, 0, y, z))

    while len(q) > 0:
        p, x, y, z = pop(q)

        if not visited[x, y, z]:
            visited[x, y, z] = True

            if x == L:
                break

            for dx, dy, dz in adj:
                x_ = x + dx
                y_ = (y + dy) % L
                z_ = (z + dz) % L
                if x_ > 0 and not visited[x_, y_, z_]:
                    push(q, (rs[i], x_, y_, z_))
                    i += 1

    return np.sum(visited[1:, :, :])

def invperc2(L):
    L = int(L)

    adj = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    visited = np.zeros((L + 1, L), dtype = bool)

    rs = npr.random((6 * (L + 1) * L,))

    i = 0
    q = []
    for y in range(L):
        q.append((0, 0, y))

    while len(q) > 0:
        p, x, y = pop(q)

        if not visited[x, y]:
            visited[x, y] = True

            if x == L:
                break

            for dx, dy in adj:
                x_ = x + dx
                y_ = (y + dy) % L
                if x_ > 0 and not visited[x_, y_]:
                    push(q, (rs[i], x_, y_))
                    i += 1

    return visited

def print_example():
    v = invperc2(1000)
    a, b = v.shape

    with open('invperc_example', 'w') as f:
        for j in range(b):
            s = ' '.join([str(v[i, j].astype(int)) for i in range(a)])
            f.write(s)
            f.write('\n')

usage=(
"""A program for simulating invasion percolation.
Usage:
    python perc.py L

The program simulates a three dimensional domain of size L x L x L
with periodic boundary conditions.
""")

def main():
    args = sys.argv
    if len(args) == 2:
        L = int(args[1])
        print(invperc(L))
    else:
        print(usage)

if __name__ == "__main__":
    main()
