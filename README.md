# cadical_py — Python wrapper for CaDiCaL (External Propagator)

Thin, Pythonic bindings to [CaDiCaL](https://github.com/arminbiere/cadical), including the **External Propagator** API so you can implement domain-specific propagation in Python while CaDiCaL does the CDCL search.


## Install
Build CaDiCaL once:
```bash
git clone https://github.com/arminbiere/cadical ~/github/cadical
cd ~/github/cadical && ./configure
make clean && make CXXFLAGS='-O3 -fPIC'
```

Install `cadical_py`:

```bash
export CADICAL=~/github/cadical
pip install .
# or dev mode
pip install -e .
```

## Quickstart

```python
from cadical_py import Solver

with Solver() as S:
    S.add_clause([+1, +2])   # (x1 ∨ x2)
    S.add_clause([-1, +3])   # (¬x1 ∨ x3)
    sat = S.solve()          # 10/20 or True/False depending on build
    print("SAT:", sat, "model:", S.model())
```

## External Propagator (Python)

Subclass `ExternalPropagator`, connect it, and observe variables:

```python
from cadical_py import Solver, ExternalPropagator

class MyProp(ExternalPropagator):
    def notify_assignment(self, lits): ...
    def cb_propagate(self) -> int: return 0
    def cb_add_reason_clause_lit(self, propagated_lit: int) -> int: return 0

with Solver() as S:
    P = MyProp()
    S.connect_external_propagator(P)
    S.add_observed_var(1)
    S.solve()
```

Key callbacks: `notify_new_decision_level`, `notify_backtrack`, `notify_assignment`, `cb_propagate`, `cb_add_reason_clause_lit`, `cb_decide`, `cb_check_found_model`, `cb_has_external_clause`, `cb_add_external_clause_lit`.

## Examples

All examples live in `examples/` and run with `python <file> ...`.  
Models are blocked by adding a clause negating the current assignment.

### 1) Model enumeration (toy CNF)
**File:** `examples/enumerate_models.py`  
Enumerates all models of `(x1 ∨ x2) ∧ (¬x1 ∨ ¬x2)` (exactly one true).
```bash
python examples/enumerate_models.py
````

### 2) Choose-k with look-ahead

**File:** `examples/ipasirup-ksubsets1.py`
External propagator enforces “exactly `k` of `n` variables true” **with look-ahead**:

* When `k` vars are true ⇒ force all remaining to false (AtMost).
* When `n-k` vars are false ⇒ force remaining to true (AtLeast).
  Counts solutions and checks against `C(n,k)`.

```bash
python examples/ipasirup-ksubsets1.py  n  k
# e.g.
python examples/ipasirup-ksubsets1.py 6 3   # expect 20
```

### 3) Choose-k (conflict-only; no look-ahead)

**File:** `examples/ipasirup-ksubsets2.py`
Same constraint, but **only detects conflicts** in `notify_assignment`:

* Abort when `k+1` trues or `(n-k)+1` falses are assigned.
* No proactive implications.
  Counts solutions and checks against `C(n,k)`.

```bash
python examples/ipasirup-ksubsets2.py  n  k
```

### 4) Unlabeled graphs via canonicality (external propagator)

**File:** `examples/ipasirup-graphs.py`
One variable per unordered vertex pair. The propagator rejects models whose
adjacency matrix is not lexicographically minimal under vertex permutations
(canonical labeling by colex order). Prints the count of unlabeled graphs and
compares small `n` to OEIS A000088.

```bash
python examples/ipasirup-graphs.py  n
# e.g.
python examples/ipasirup-graphs.py 6
```

