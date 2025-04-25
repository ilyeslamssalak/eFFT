import numpy as np

class Stimulus:
    def __init__(self, row: int, col: int, state: bool = True):
        self.row = row
        self.col = col
        self.state = state

    def __eq__(self, other):
        return isinstance(other, Stimulus) and self.row == other.row and self.col == other.col

    def __ne__(self, other):
        return not self.__eq__(other)

class Stimuli(list):
    def filter(self):
        # sort by row, then col, then state (True before False)
        self.sort(key=lambda p: (p.row, p.col, not p.state))
        # unique by (row, col)
        unique = []
        seen = set()
        for p in self:
            if (p.row, p.col) not in seen:
                unique.append(p)
                seen.add((p.row, p.col))
        self.clear()
        self.extend(unique)

    def set_state(self, state: bool):
        for p in self:
            p.state = state

class EFFT:
    def __init__(self, N: int):
        if N & (N - 1) != 0:
            raise ValueError("N must be a power of 2")
        self.N = N
        self.LOG2_N = int(np.log2(N))
        # tree_ is a list of lists of matrices at each log-level
        self.tree = [ [] for _ in range(self.LOG2_N + 1) ]
        # precompute twiddle factors shape (N, N+1)
        minus_two_pi = -2 * np.pi
        self.twiddle = np.zeros((N, N+1), dtype=np.complex64)
        for i in range(N):
            for n in range(N+1):
                self.twiddle[i, n] = np.exp(1j * (minus_two_pi * i / n)) if n > 0 else 1.0

    def initialize(self, x: np.ndarray = None, offset: int = 0):
        # x should be N x N complex array
        if x is None:
            x = np.zeros((self.N, self.N), dtype=np.complex64)
        n = x.shape[0]
        if n == 1:
            # base case
            self.tree[0].append(x.copy())
            return
        ndiv2 = n // 2
        idx = int(np.log2(ndiv2))
        # split into quadrants
        s00 = x[0::2, 0::2]
        s01 = x[0::2, 1::2]
        s10 = x[1::2, 0::2]
        s11 = x[1::2, 1::2]
        # recurse
        self.initialize(s00,     4*offset)
        self.initialize(s01, 4*offset+1)
        self.initialize(s10, 4*offset+2)
        self.initialize(s11, 4*offset+3)
        # combine
        x00 = self.tree[idx][offset]
        x01 = self.tree[idx][offset+1]
        x10 = self.tree[idx][offset+2]
        x11 = self.tree[idx][offset+3]
        Nn = self.N * n
        # perform butterfly
        for i in range(ndiv2):
            for j in range(ndiv2):
                tu = self.twiddle[j, Nn] * x01[i, j]
                td = self.twiddle[i+j, Nn] * x11[i, j]
                ts = self.twiddle[i, Nn] * x10[i, j]
                a = x00[i, j] + tu
                b = x00[i, j] - tu
                c = ts + td
                d = ts - td
                x[i, j] = a + c
                x[i, j+ndiv2] = b + d
                x[i+ndiv2, j] = a - c
                x[i+ndiv2, j+ndiv2] = b - d
        # store result
        self.tree[idx+1].append(x.copy())

    def update(self, stimuli):
        if isinstance(stimuli, Stimulus):
            return self._update_matrix(self.tree[self.LOG2_N][0], stimuli)
        elif isinstance(stimuli, Stimuli):
            changed = False
            for p in stimuli:
                changed = self._update_matrix(self.tree[self.LOG2_N][0], p) or changed
            return changed
        else:
            raise TypeError("update expects Stimulus or Stimuli")

    def _update_matrix(self, x: np.ndarray, p: Stimulus, offset: int = 0) -> bool:
        n = x.shape[0]
        if n == 1:
            prev = x[0, 0]
            x[0, 0] = 1.0+0j if p.state else 0.0+0j
            return prev != x[0, 0]
        ndiv2 = n // 2
        idx = int(np.log2(ndiv2))
        # select quadrant
        sub_row = p.row >> 1
        sub_col = p.col >> 1
        if p.row & 1:
            if p.col & 1:
                child_offset = 4*offset + 3
            else:
                child_offset = 4*offset + 2
        else:
            if p.col & 1:
                child_offset = 4*offset + 1
            else:
                child_offset = 4*offset
        # recurse
        changed = self._update_matrix(
            self.tree[idx][child_offset],
            Stimulus(sub_row, sub_col, p.state),
            child_offset
        )
        if changed:
            # fetch updated quadrants
            x00 = self.tree[idx][offset]
            x01 = self.tree[idx][offset+1]
            x10 = self.tree[idx][offset+2]
            x11 = self.tree[idx][offset+3]
            Nn = self.N * n
            for i in range(ndiv2):
                for j in range(ndiv2):
                    tu = self.twiddle[j, Nn] * x01[i, j]
                    td = self.twiddle[i+j, Nn] * x11[i, j]
                    ts = self.twiddle[i, Nn] * x10[i, j]
                    a = x00[i, j] + tu
                    b = x00[i, j] - tu
                    c = ts + td
                    d = ts - td
                    x[i, j] = a + c
                    x[i, j+ndiv2] = b + d
                    x[i+ndiv2, j] = a - c
                    x[i+ndiv2, j+ndiv2] = b - d
        return changed

    def get_fft(self) -> np.ndarray:
        return self.tree[self.LOG2_N][0]
