get_ipython().run_line_magic('load_ext', 'Cython')


import numpy as np
from PIL import Image
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt

def random_spin_field(N, M):
    return np.random.choice([-1, 1], size=(N, M))

def display_spin_field(field):
    return Image.fromarray(np.uint8((field + 1) * 0.5 * 255))  # 0 ... 255

get_ipython().run_cell_magic('cython', '', '\ncimport cython\n\nimport numpy as np\ncimport numpy as np\n\nfrom libc.math cimport exp\nfrom libc.stdlib cimport rand\ncdef extern from "limits.h":\n    int RAND_MAX\n\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ndef cy_ising_step(np.int64_t[:, :] field, float beta=0.4):\n    cdef int N = field.shape[0]\n    cdef int M = field.shape[1]\n    cdef int n_offset, m_offset, n, m\n    for n_offset in range(2):\n        for m_offset in range(2):\n            for n in range(n_offset, N, 2):\n                for m in range(m_offset, M, 2):\n                    _cy_ising_update(field, n, m, beta)\n    return np.array(field)\n\n\n@cython.boundscheck(False)\n@cython.wraparound(False)\ncdef _cy_ising_update(np.int64_t[:, :] field, int n, int m, float beta):\n    cdef int total = 0\n    cdef int N = field.shape[0]\n    cdef int M = field.shape[1]\n    cdef int i, j\n    for i in range(n-1, n+2):\n        for j in range(m-1, m+2):\n            if i == n and j == m:\n                continue\n            total += field[i % N, j % M]\n    cdef float dE = 2 * field[n, m] * total\n    if dE <= 0:\n        field[n, m] *= -1\n    elif exp(-dE * beta) * RAND_MAX > rand():\n        field[n, m] *= -1')




images = [random_spin_field(200, 200)]
for i in range(100):
    images.append(cy_ising_step(images[-1].copy(), beta=0.4))
    sns.heatmap(images[i], cmap=plt.cm.inferno_r)
    plt.axis('off')
    plt.savefig('./simulation/'+'step'+str(i)+'.png')
    plt.clf()
