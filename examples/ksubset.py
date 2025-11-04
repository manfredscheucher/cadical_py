# choose_k_prop.py
from collections import deque
from typing import List, Tuple, Dict, Set
import cadipy

def binom(n: int, k: int) -> int:
    if k > n: return 0
    k = min(k, n - k)
    r = 1
    for i in range(1, k + 1):
        r = (r * (n - k + i)) // i
    return r

class ChooseKPropPy:
    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k
        self.S = None

        # Level-Stacks / States
        self.level = 0
        self.assigned_at_level: List[Set[int]] = [set()]
        self.reason_lits_at_level: List[Set[int]] = [set()]
        self.pos: Set[int] = set()
        self.neg: Set[int] = set()

        # Propagation
        self.prop_queue: deque[int] = deque()
        self.reason_of: Dict[int, List[int]] = {}

        # External clauses
        self._cur_ext: List[int] = []
        self.ext_clauses: deque[List[int]] = deque()

        # Stats
        self.solutions = 0

    # --------- Helpers ----------
    def is_unassigned_var(self, v: int) -> bool:
        return (v not in self.pos) and (v not in self.neg)

    def enqueue_implication(self, lit: int, reason: List[int]) -> None:
        if lit not in self.reason_of:
            self.prop_queue.append(lit)
            self.reason_of[lit] = list(reason)
            self.reason_lits_at_level[self.level].add(lit)

    def _at_most_k_propagate(self) -> None:
        if len(self.pos) != self.k:
            return
        prefix = sorted([-t for t in self.pos])
        for v in range(1, self.n + 1):
            if self.is_unassigned_var(v):
                clause = prefix + [-v]
                self.enqueue_implication(-v, clause)

    def _at_least_k_propagate(self) -> None:
        if len(self.neg) != (self.n - self.k):
            return
        prefix = sorted([+t for t in self.neg])
        for v in range(1, self.n + 1):
            if self.is_unassigned_var(v):
                clause = prefix + [+v]
                self.enqueue_implication(+v, clause)

    # --------- C++ callback surface (called from C++ trampolin) ----------
    def notify_new_decision_level(self) -> None:
        self.level += 1
        self.assigned_at_level.append(set())
        self.reason_lits_at_level.append(set())

    def notify_backtrack(self, new_level: int) -> None:
        while self.level > new_level:
            for lit in self.assigned_at_level[self.level]:
                if lit > 0: self.pos.discard(lit)
                else:       self.neg.discard(-lit)
            self.assigned_at_level.pop()
            for lit in self.reason_lits_at_level[self.level]:
                self.reason_of.pop(lit, None)
            self.reason_lits_at_level.pop()
            self.level -= 1
        self.prop_queue.clear()

    def notify_assignment(self, lits: List[int]) -> None:
        for lit in lits:
            self.assigned_at_level[self.level].add(lit)
            if lit > 0: self.pos.add(lit)
            else:       self.neg.add(-lit)
        self._at_most_k_propagate()
        self._at_least_k_propagate()

    def cb_propagate(self) -> int:
        return self.prop_queue.popleft() if self.prop_queue else 0

    def cb_add_reason_clause_lit(self, propagated_lit: int) -> int:
        cls = self.reason_of.get(propagated_lit)
        if not cls:
            return 0
        lit = cls.pop()  # reverse order, like in your C++
        if not cls:
            # after last literal, we remove the entry;
            # CaDiCaL will call again and get 0 to terminate
            self.reason_of.pop(propagated_lit, None)
        return lit

    def cb_decide(self) -> int:
        return 0

    def cb_check_found_model(self, model: List[int]) -> bool:
        t = 0
        block = []
        # model is ±v at index v-1
        for v in range(1, self.n + 1):
            val = model[v - 1]
            if val > 0:
                t += 1
                block.append(-v)
            else:
                block.append(+v)
        if t != self.k:
            self.ext_clauses.append(block)
            return False
        self.solutions += 1
        self.ext_clauses.append(block)
        return True

    def cb_has_external_clause(self) -> Tuple[bool, bool]:
        # returns (has_clause, is_forgettable)
        return (len(self.ext_clauses) > 0, True)

    def cb_add_external_clause_lit(self) -> int:
        if not self._cur_ext:
            if not self.ext_clauses:
                return 0
            self._cur_ext = list(self.ext_clauses.popleft())
        if not self._cur_ext:
            return 0
        lit = self._cur_ext.pop()
        if not self._cur_ext:
            # Next call will yield 0 to terminate the clause.
            return 0
        return lit

# ---- runner (example) ----
def run_choose_k(n: int, k: int) -> None:
    solver = cadipy.Solver()
    prop = ChooseKPropPy(n, k)
    solver.connect_external_propagator(prop)
    for v in range(1, n + 1):
        solver.add_observed_var(v)
    while True:
        res = solver.solve()   # 10=SAT, 20=UNSAT
        if res == 20:
            break
        if res != 10:
            print("UNKNOWN")
            return
        # on SAT: cb_check_found_model handles counting + blocking

    print(f"{prop.solutions} solutions; expected C({n},{k}) = {binom(n,k)}")



if __name__ == "__main__":
    import sys
    n = int(sys.argv[1])
    k = int(sys.argv[2])
    run_choose_k(n,k)
