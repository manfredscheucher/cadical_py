# examples/ipasirup-graphs.py — same logic as PySAT version, but aligned with ksubsets1/2 style
from itertools import combinations
from sys import argv
from collections import deque
from typing import List, Set, Dict, Tuple
from cadical_py import Solver, ExternalPropagator

n = int(argv[1])

vars_ids: Dict[int, Tuple[int,int]] = {i: I for (i, I) in enumerate(combinations(range(n), 2), 1)}
rev_vars: Dict[Tuple[int,int], int] = {I: i for (i, I) in enumerate(combinations(range(n), 2), 1)}
for i, j in combinations(range(n), 2):
    rev_vars[j, i] = rev_vars[i, j]
assert all(rev_vars[vars_ids[i]] == i for i in vars_ids)

check_model_calls = 0
propagate_calls = 0


def is_symmetric(A):
    n_ = len(A)
    return all(A[r][c] == A[c][r] for c in range(n_) for r in range(n_))

def permute(A, perm):
    assert all(x in range(len(A)) for x in perm)
    return [[A[r][c] for c in perm] for r in perm]

def combinations_colex_ordered(L, r):
    return list(reversed([tuple(reversed(I)) for I in combinations(reversed(L), r)]))

def fingerprint(A):
    return [A[i][j] for i, j in combinations_colex_ordered(range(len(A)), 2)]

def replace(L, oldvalue, newvalue):
    return [(newvalue if x == oldvalue else x) for x in L]

def minCheck(A, perm=tuple()):
    if not perm:
        assert is_symmetric(A)
    k = len(perm)
    n_ = len(A)
    A_ = permute(A, range(k))
    B_ = permute(A, perm)
    A_fp = replace(fingerprint(A_), None, 0)
    B_fp = replace(fingerprint(B_), None, 1)
    assert len(A_fp) == len(B_fp)

    if B_fp > A_fp:
        return

    if k == n_:
        if B_fp < A_fp:
            order_A = combinations_colex_ordered(range(k), 2)
            order_B = combinations_colex_ordered(perm, 2)
            assert all(y == (perm[x[0]], perm[x[1]]) for x, y in zip(order_A, order_B))

            must_be_positive = set()
            must_be_negative = set()
            found = False
            for l in range(len(B_fp)):
                if B_fp[l] == 0:
                    must_be_positive.add(rev_vars[order_B[l]])
                if A_fp[l] == 1:
                    must_be_negative.add(rev_vars[order_A[l]])
                if B_fp[l] != A_fp[l]:
                    assert (B_fp[l] == 0 and A_fp[l] == 1)
                    yield perm, must_be_positive, must_be_negative
                    found = True
                    break
            assert found
    else:
        assert k < n_
        for v in range(n_):
            if v not in perm:
                yield from minCheck(A, perm + (v,))


class GraphPropagator(ExternalPropagator):
    def __init__(self, n_):
        self.n = n_
        self.level = 0

        self.assigned_at_level: List[Set[int]] = [set()]
        self.reason_lits_at_level: List[Set[int]] = [set()]
        self.pos: Set[int] = set()
        self.neg: Set[int] = set()

        self.q: deque[int] = deque()
        self.reason_of: Dict[int, List[int]] = {}

        self.ext: deque[List[int]] = deque()
        self.cur: List[int] | None = None

    def setup_observe(self, solver: Solver):
        for v in vars_ids:
            solver.add_observed_var(v)

    def adjMatrix(self):
        A = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for i, (r, c) in vars_ids.items():
            if i in self.pos:
                A[r][c] = A[c][r] = 1
            elif i in self.neg:
                A[r][c] = A[c][r] = 0
            else:
                A[r][c] = A[c][r] = None
        return A

    # ---- trail notifications ----
    def notify_new_decision_level(self):
        self.level += 1
        self.assigned_at_level.append(set())
        self.reason_lits_at_level.append(set())

    def notify_backtrack(self, to):
        while self.level > to:
            for lit in self.assigned_at_level[self.level]:
                if lit > 0: self.pos.discard(lit)
                else:       self.neg.discard(-lit)
            self.assigned_at_level.pop()

            for lit in self.reason_lits_at_level[self.level]:
                self.reason_of.pop(lit, None)
            self.reason_lits_at_level.pop()
            self.level -= 1

        self.q.clear()
        self.cur = None

    # ---- partial assignment handling (same logic) ----
    def notify_assignment(self, lits):
        for lit in lits:
            self.assigned_at_level[self.level].add(lit)
            if lit > 0: self.pos.add(lit)
            else:       self.neg.add(-lit)

        for _, must_pos, must_neg in minCheck(self.adjMatrix()):
            assert must_neg.issubset(self.pos)
            assert must_pos.issubset(self.neg)
            clause = [ +x for x in must_pos ] + [ -x for x in must_neg ]
            lit = clause[0]
            if lit not in self.reason_of:
                self.q.append(lit)
                self.reason_of[lit] = list(clause)
                self.reason_lits_at_level[self.level].add(lit)
            return

    # ---- propagation & reasons (unified style) ----
    def cb_propagate(self) -> int:
        return self.q.popleft() if self.q else 0

    def cb_add_reason_clause_lit(self, propagated_lit: int) -> int:
        cls = self.reason_of.get(propagated_lit)
        if not cls: return 0
        lit = cls.pop()
        if not cls: self.reason_of.pop(propagated_lit, None)
        return lit  # next call will return 0

    # ---- model check & external clauses (enumeration) ----
    def cb_check_found_model(self, model: List[int]) -> bool:
        global check_model_calls
        check_model_calls += 1

        for _ in minCheck(self.adjMatrix()):
            print("this should never happen! on_assignment should filter all invalid configurations!")
            self.ext.append([-l for l in model if l != 0])
            return False

        # accept; block to enumerate
        self.ext.append([-l for l in model if l != 0])
        return True

    def cb_has_external_clause(self):
        return (len(self.ext) > 0, True)

    def cb_add_external_clause_lit(self) -> int:
        if not self.cur:
            if not self.ext: return 0
            self.cur = list(self.ext.popleft())
        lit = self.cur.pop()
        if not self.cur:
            self.cur = None
        return lit

    def cb_decide(self) -> int:
        return 0


solutions = []
with Solver() as s:
    P = GraphPropagator(n)
    s.connect_external_propagator(P)
    P.setup_observe(s)

    # empty CNF; propagator drives the search
    while True:
        r = s.solve()
        if r == 20: break
        if r != 10:
            print("UNKNOWN")
            break

        model = s.model()
        A = [[0 for _ in range(n)] for _ in range(n)]
        for i in vars_ids:
            r0, c0 = vars_ids[i]
            A[r0][c0] = A[c0][r0] = 1
        solutions.append(A)
        s.add_clause([-l for l in model if l != 0])

print(f"{len(solutions)} solutions")
if len(argv) > 2:
    print(solutions)

expected = [1, 1, 2, 4, 11, 34, 156, 1044, 12346, 274668, 12005168, 1018997864, 165091172592]  # OEIS A000088
assert len(solutions) == expected[n]

print(f"check_model_calls: {check_model_calls}")
print(f"propagate_calls: {propagate_calls}")
