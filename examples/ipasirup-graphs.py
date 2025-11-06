# examples/ipasirup-graphs.py — same logic as PySAT version, using cadical_py
from itertools import combinations
from sys import argv

from cadical_py import Solver, ExternalPropagator

n = int(argv[1])

vars_ids = {i: I for (i, I) in enumerate(combinations(range(n), 2), 1)}
rev_vars = {I: i for (i, I) in enumerate(combinations(range(n), 2), 1)}
for i, j in combinations(range(n), 2):
    rev_vars[j, i] = rev_vars[i, j]  # undirected graph = symmetric adjacency matrix
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
    A_ = permute(A, range(k))  # restrict to first k rows and columns
    B_ = permute(A, perm)      # restrict to k rows and columns from permutation
    A_fp = fingerprint(A_)
    B_fp = fingerprint(B_)
    A_fp = replace(A_fp, None, 0)  # unset in original are assumed zeros (best case)
    B_fp = replace(B_fp, None, 1)  # unset in permuted are assumed ones (worst case)
    assert len(A_fp) == len(B_fp)

    if B_fp > A_fp:
        return

    if k == n_:
        if B_fp < A_fp:  # omit identity
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
        super().__init__()
        self.n = n_
        self.level = 0

        self.assigned_at_level = [set()]  # for each level: list of literals assigned
        self.assigned_positive = set()
        self.assigned_negative = set()

        self.reason = {}               # reason for propagated literals: lit -> list[int]
        self.reason_at_level = [set()] # per level (to clear on backtrack)
        self.queue = []                # literals queued for propagation
        self.pending = []              # external clause from check_model()

        self._cur_reason = {}          # lit -> current remaining lits to stream
        self._cur_ext = []             # current external clause being streamed

    def adjMatrix(self):
        A = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for i in vars_ids.keys():
            r, c = vars_ids[i]
            if i in self.assigned_positive:
                A[r][c] = A[c][r] = 1
            elif i in self.assigned_negative:
                A[r][c] = A[c][r] = 0
            else:
                A[r][c] = A[c][r] = None
        return A

    # --- cadical_py -> observe variables
    def setup_observe(self, solver):
        for v in vars_ids:
            solver.add_observed_var(v)

    # --- CaDiCaL ExternalPropagator callbacks (names adapted, logic unchanged) ---

    def notify_new_decision_level(self):
        self.level += 1
        self.assigned_at_level.append(set())
        self.reason_at_level.append(set())
        assert len(self.assigned_at_level) == self.level + 1
        assert len(self.reason_at_level) == self.level + 1

    def notify_backtrack(self, to):
        while self.level > to:
            for lit in self.assigned_at_level[self.level]:
                if lit > 0:
                    self.assigned_positive.discard(lit)
                else:
                    self.assigned_negative.discard(-lit)
            self.assigned_at_level.pop()

            for lit in self.reason_at_level[self.level]:
                self.reason.pop(lit, None)
                self._cur_reason.pop(lit, None)
            self.reason_at_level.pop()

            self.level -= 1
        self.queue.clear()
        self._cur_ext = []

    # check partial assignment
    def notify_assignment(self, lits):
        for lit in lits:
            assert lit not in self.assigned_positive and lit not in self.assigned_negative
            self.assigned_at_level[self.level].add(lit)
            if lit > 0:
                self.assigned_positive.add(lit)
            else:
                self.assigned_negative.add(-lit)

        for better_perm, must_be_positive, must_be_negative in minCheck(self.adjMatrix()):
            assert (must_be_negative.issubset(self.assigned_positive))
            assert (must_be_positive.issubset(self.assigned_negative))
            conflict_clause = [+x for x in must_be_positive] + [-x for x in must_be_negative]
            impl = conflict_clause[0]
            self.queue.append(impl)
            self.reason[impl] = list(conflict_clause)
            self.reason_at_level[self.level].add(impl)
            return

    # return one implied lit (0 if none)
    def cb_propagate(self):
        if not self.queue:
            return 0
        global propagate_calls
        propagate_calls += 1
        return self.queue.pop(0)

    # stream reason clause for a propagated literal
    def cb_add_reason_clause_lit(self, propagated_lit):
        if propagated_lit not in self._cur_reason:
            cls = self.reason.pop(propagated_lit, [])
            self._cur_reason[propagated_lit] = list(reversed(cls))  # stream via pop()
        cur = self._cur_reason[propagated_lit]
        if not cur:
            self._cur_reason.pop(propagated_lit, None)
            return 0
        return cur.pop()

    # check full assignment
    def cb_check_found_model(self, model):
        global check_model_calls
        check_model_calls += 1
        
        for _ in minCheck(self.adjMatrix()):
            print("this should never happen! on_assignment should filter all invalid configurations!")
            self.pending = [-l for l in model]
            return False

        return True

    # external clause availability
    def cb_has_external_clause(self):
        return (len(self.pending) > 0, True)

    # stream external clause lits (0 terminates)
    def cb_add_external_clause_lit(self):
        if not self._cur_ext:
            if not self.pending:
                return 0
            self._cur_ext = list(reversed(self.pending))
            self.pending = []
        lit = self._cur_ext.pop()
        if not self._cur_ext:
            self._cur_ext = []
            return 0
        return lit

    def cb_decide(self):
        return 0


solutions = []
with Solver() as s:
    p = GraphPropagator(n)
    s.connect_external_propagator(p)
    p.setup_observe(s)

    # empty CNF
    while True:
        res = s.solve()
        if res == 20:
            break
        if res != 10:
            print("UNKNOWN")
            break

        model = s.model()
        A = [[0 for _ in range(n)] for _ in range(n)]
        for i in vars_ids:
            r, c = vars_ids[i]
            A[r][c] = A[c][r] = 1
        solutions.append(A)
        s.add_clause([-l for l in model if l != 0])

print(f"{len(solutions)} solutions")
if len(argv) > 2:
    print(solutions)

expected_count = [1, 1, 2, 4, 11, 34, 156, 1044, 12346, 274668, 12005168, 1018997864, 165091172592]  # OEIS A000088
assert len(solutions) == expected_count[n]

print(f"check_model_calls: {check_model_calls}")
print(f"propagate_calls: {propagate_calls}")
