# ksubsets.py
# Enumerates all {0,1}^n assignments with exactly k TRUEs using a Python
# ExternalPropagator. Blocking is streamed via external-clause callbacks
# (no direct solver.add(...) needed), identical pattern to graphsym.

from collections import deque
from typing import List
import sys

import cadical_py as cpy  # angepasst an dein Paket

def binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    r = 1
    for i in range(1, k + 1):
        r = (r * (n - k + i)) // i
    return r


class ChooseKPropPy:
    """
    Propagator that enforces exactly k TRUE among n variables 1..n.
    - AtMost(k): once k vars are TRUE, all remaining must be FALSE.
    - AtLeast(k): once n-k vars are FALSE, all remaining must be TRUE.
    - Enumerates models and pushes blocking clauses via external callbacks.
    """
    def __init__(self, n: int, k: int):
        self.S = None
        self.n = int(n)
        self.k = int(k)

        # decision-level stacks
        self.level = 0
        self.assigned_at_level = [set()]   # signed lits per level
        self.reason_lits_at_level = [set()]

        # current assignment (by var id)
        self.pos = set()   # vars set TRUE
        self.neg = set()   # vars set FALSE

        # propagation & reasons
        self.prop_queue: deque[int] = deque()
        self.reason_of: dict[int, List[int]] = {}  # implied lit -> reason clause (stack)

        # external clauses (blocking etc.)
        self._ext_clauses: deque[List[int]] = deque()
        self._cur_ext_clause: List[int] | None = None

        # stats
        self.solutions = 0
        self.propagate_calls = 0

    # ---- helpers ----
    def is_unassigned_var(self, v: int) -> bool:
        return (v not in self.pos) and (v not in self.neg)

    def enqueue_implication(self, lit: int, reason: List[int]) -> None:
        if lit not in self.reason_of:
            self.prop_queue.append(lit)
            # store reason as stack for streaming
            self.reason_of[lit] = list(reason)[::-1]
            self.reason_lits_at_level[self.level].add(lit)

    def at_most_k_propagate(self) -> None:
        if len(self.pos) != self.k:
            return
        # clause template: (¬t1 ∨ ¬t2 ∨ ... ∨ ¬tk ∨ ¬v)
        prefix = sorted(-t for t in self.pos)
        for v in range(1, self.n + 1):
            if self.is_unassigned_var(v):
                self.enqueue_implication(-v, prefix + [-v])

    def at_least_k_propagate(self) -> None:
        if len(self.neg) != (self.n - self.k):
            return
        # clause template: (f1 ∨ f2 ∨ ... ∨ fk' ∨ v) with f_i are the FALSE vars
        prefix = sorted(+t for t in self.neg)
        for v in range(1, self.n + 1):
            if self.is_unassigned_var(v):
                self.enqueue_implication(+v, prefix + [+v])

    # ---- ExternalPropagator (duck-typed) ----
    def init(self, solver):
        self.S = solver

    def notify_new_decision_level(self):
        self.level += 1
        self.assigned_at_level.append(set())
        self.reason_lits_at_level.append(set())

    def notify_backtrack(self, new_level: int):
        while self.level > new_level:
            for lit in self.assigned_at_level[self.level]:
                if lit > 0:
                    self.pos.discard(lit)
                else:
                    self.neg.discard(-lit)
            self.assigned_at_level.pop()
            for lit in self.reason_lits_at_level[self.level]:
                self.reason_of.pop(lit, None)
            self.reason_lits_at_level.pop()
            self.level -= 1
        self.prop_queue.clear()
        self._cur_ext_clause = None

    def notify_assignment(self, lits: List[int]):
        for lit in lits:
            self.assigned_at_level[self.level].add(lit)
            if lit > 0:
                self.pos.add(lit)
            else:
                self.neg.add(-lit)
        self.at_most_k_propagate()
        self.at_least_k_propagate()

    def cb_propagate(self) -> int:
        if not self.prop_queue:
            return 0
        self.propagate_calls += 1
        return self.prop_queue.popleft()

    def cb_add_reason_clause_lit(self, propagated_lit: int) -> int:
        cls = self.reason_of.get(propagated_lit)
        if not cls:
            self.reason_of.pop(propagated_lit, None)
            return 0
        lit = cls.pop()
        if not cls:
            self.reason_of.pop(propagated_lit, None)
        return lit

    def cb_decide(self) -> int:
        return 0

    def cb_check_found_model(self, model: List[int]) -> bool:
        # model is ±v at index v-1
        t = sum(1 for val in model[:self.n] if val > 0)
        # block the full cube over 1..n either way
        block = [-(i + 1) if model[i] > 0 else (i + 1) for i in range(self.n)]
        self._ext_clauses.append(block)
        if t == self.k:
            self.solutions += 1
            return True   # accept model
        return False       # reject (forces adding our blocking clause)

    # external clause streaming (blocking)
    def cb_has_external_clause(self) -> tuple[bool, bool]:
        return (len(self._ext_clauses) > 0, True)

    def cb_add_external_clause_lit(self) -> int:
        if self._cur_ext_clause is None:
            if not self._ext_clauses:
                return 0
            self._cur_ext_clause = list(self._ext_clauses.popleft())[::-1]
        if not self._cur_ext_clause:
            self._cur_ext_clause = None
            return 0
        return self._cur_ext_clause.pop()


def run_ksubset(n: int, k: int) -> tuple[int, int]:
    """
    Returns (solutions, propagate_calls)
    """
    if not (0 <= k <= n):
        raise ValueError("require 0 <= k <= n")

    S = cpy.Solver()
    P = ChooseKPropPy(n, k)
    S.connect_external_propagator(P)
    for v in range(1, n + 1):
        S.add_observed_var(v)

    while True:
        code = S.solve()  # 10 = SAT, 20 = UNSAT
        if code == 20:
            break
        if code != 10:
            raise RuntimeError("UNKNOWN solver status")
        # on SAT: cb_check_found_model() handled counting + blocking

    assert(P.solutions == binom(n,k))
    return P.solutions, P.propagate_calls


if __name__ == "__main__":
    if len(sys.argv) == 3:
        n = int(sys.argv[1])
        k = int(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} n [k]")
        sys.exit(1)

    sols, calls = run_ksubset(n, k)
    print(f"{sols} solutions")
    print(f"propagate_calls: {calls}")
