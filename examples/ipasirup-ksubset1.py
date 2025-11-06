# examples/ipasirup-ksubsets1.py — enumerate exactly k of n true (with look-ahead)

from __future__ import annotations
from collections import deque
from typing import List, Set, Dict
from cadical_py import Solver, ExternalPropagator

def binom(n: int, k: int) -> int:
    k = min(k, n - k); r = 1
    for i in range(1, k + 1): r = (r * (n - k + i)) // i
    return r

class KSubset(ExternalPropagator):
    def __init__(self, n: int, k: int):
        self.n, self.k = n, k
        self.level = 0
        self.pos: Set[int] = set(); self.neg: Set[int] = set()
        self.ass_l: List[Set[int]] = [set()]
        self.reas_l: List[Set[int]] = [set()]
        self.q: deque[int] = deque()
        self.reason_of: Dict[int, List[int]] = {}
        self.block: deque[List[int]] = deque(); self.cur: List[int] | None = None
        self.count = 0

    def notify_new_decision_level(self): 
        self.level += 1; self.ass_l.append(set()); self.reas_l.append(set())

    def notify_backtrack(self, new_level: int):
        while self.level > new_level:
            for lit in self.ass_l[self.level]:
                (self.pos.discard(lit) if lit > 0 else self.neg.discard(-lit))
            self.ass_l.pop()
            for lit in self.reas_l[self.level]: self.reason_of.pop(lit, None)
            self.reas_l.pop(); self.level -= 1
        self.q.clear(); self.cur = None

    def notify_assignment(self, lits: List[int]):
        for lit in lits:
            self.ass_l[self.level].add(lit)
            (self.pos.add(lit) if lit > 0 else self.neg.add(-lit))
        # AtMost(k): if k vars true, force remaining false
        if len(self.pos) == self.k:
            pref = sorted(-v for v in self.pos)
            for v in range(1, self.n + 1):
                if v not in self.pos and v not in self.neg:
                    lit, cls = -v, (pref + [-v])
                    if lit not in self.reason_of:
                        self.q.append(lit); self.reason_of[lit] = cls; self.reas_l[self.level].add(lit)
        # AtLeast(k): if (n-k) vars false, force remaining true
        if len(self.neg) == self.n - self.k:
            pref = sorted(+v for v in self.neg)
            for v in range(1, self.n + 1):
                if v not in self.pos and v not in self.neg:
                    lit, cls = +v, (pref + [+v])
                    if lit not in self.reason_of:
                        self.q.append(lit); self.reason_of[lit] = cls; self.reas_l[self.level].add(lit)

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
        if t != self.k: self.block.append(block); return False
        self.count += 1; self.block.append(block); return True

    def cb_has_external_clause(self): 
        return (len(self.block) > 0, True)

    # FIX 2: return the popped literal, clear state; next call returns 0
    def cb_add_external_clause_lit(self) -> int:
        if not self.cur:
            if not self.block: return 0
            self.cur = list(self.block.popleft())
        lit = self.cur.pop()
        if not self.cur: self.cur = None
        return lit

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3: print(f"Usage: {sys.argv[0]} n k"); sys.exit(0)
    n, k = map(int, sys.argv[1:])
    with Solver() as S:
        P = KSubset(n, k); S.connect_external_propagator(P)
        for v in range(1, n + 1): S.add_observed_var(v)
        while True:
            r = S.solve()
            if r == 20: break
            if r != 10: print("UNKNOWN"); break
        exp = binom(n, k)
        print(f"solutions: {P.count}, expected C({n},{k}) = {exp}" + ("" if P.count == exp else "  (mismatch!)"))
