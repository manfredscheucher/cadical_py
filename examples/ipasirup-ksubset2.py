# examples/ipasirup-ksubsets2.py — exactly k of n true, NO look-ahead, checks C(n,k)
from __future__ import annotations
from collections import deque
from typing import List, Set, Dict
from cadical_py import Solver, ExternalPropagator

def binom(n: int, k: int) -> int:
    k = min(k, n - k); r = 1
    for i in range(1, k + 1): r = (r * (n - k + i)) // i
    return r

class KSubsetNoLookahead(ExternalPropagator):
    def __init__(self, n: int, k: int):
        self.n, self.k = n, k
        self.level = 0
        self.pos: Set[int] = set(); self.neg: Set[int] = set()
        self.ass:  List[Set[int]] = [set()]
        self.reas: List[Set[int]] = [set()]
        self.q: deque[int] = deque()
        self.reason_of: Dict[int, List[int]] = {}
        self.blocks: deque[List[int]] = deque(); self.cur: List[int] | None = None
        self.count = 0

    def notify_new_decision_level(self):
        self.level += 1; self.ass.append(set()); self.reas.append(set())

    def notify_backtrack(self, new_level: int):
        while self.level > new_level:
            for lit in self.ass[self.level]:
                (self.pos.discard(lit) if lit > 0 else self.neg.discard(-lit))
            self.ass.pop()
            for lit in self.reas[self.level]: self.reason_of.pop(lit, None)
            self.reas.pop(); self.level -= 1
        self.q.clear(); self.cur = None

    def notify_assignment(self, lits: List[int]):
        for lit in lits:
            self.ass[self.level].add(lit)
            (self.pos.add(lit) if lit > 0 else self.neg.add(-lit))

        # conflict-only checks
        if len(self.pos) == self.k + 1:              # AtMost(k) violated
            clause = sorted(-v for v in self.pos)    # size k+1
            t = next(iter(self.pos))
            lit = -t
        elif len(self.neg) == self.n - self.k + 1:   # AtLeast(k) violated
            clause = sorted(+v for v in self.neg)    # size n-k+1
            f = next(iter(self.neg))
            lit = +f
        else:
            return

        if lit not in self.reason_of:
            self.q.append(lit); self.reason_of[lit] = clause; self.reas[self.level].add(lit)

    def cb_propagate(self) -> int:
        return self.q.popleft() if self.q else 0

    def cb_add_reason_clause_lit(self, propagated_lit: int) -> int:
        cls = self.reason_of.get(propagated_lit)
        if not cls: return 0
        lit = cls.pop()                   # <-- give last literal too
        if not cls: self.reason_of.pop(propagated_lit, None)
        return lit                        # next call will return 0

    def cb_check_found_model(self, model: List[int]) -> bool:
        t = sum(1 for v in range(1, self.n + 1) if model[v-1] > 0)
        block = [(-v if model[v-1] > 0 else +v) for v in range(1, self.n + 1)]
        if t != self.k: self.blocks.append(block); return False
        self.count += 1; self.blocks.append(block); return True

    def cb_has_external_clause(self): return (len(self.blocks) > 0, True)

    def cb_add_external_clause_lit(self) -> int:
        if not self.cur:
            if not self.blocks: return 0
            self.cur = list(self.blocks.popleft())
        lit = self.cur.pop()              # <-- give last literal too
        if not self.cur: self.cur = None  # next call will return 0
        return lit

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3: print(f"Usage: {sys.argv[0]} n k"); sys.exit(0)
    n, k = map(int, sys.argv[1:])
    with Solver() as S:
        P = KSubsetNoLookahead(n, k); S.connect_external_propagator(P)
        for v in range(1, n + 1): S.add_observed_var(v)
        while True:
            r = S.solve()
            if r == 20: break
            if r != 10: print("UNKNOWN"); break
        exp = binom(n, k)
        print(f"solutions: {P.count}, expected C({n},{k}) = {exp}" + ("" if P.count == exp else "  (mismatch!)"))
