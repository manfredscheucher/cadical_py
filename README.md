# CaDiCaL bauen (einmalig), wichtig: -fPIC
cd ~/github/cadical
./configure
make clean
make CXXFLAGS='-O3 -fPIC'

# cadipy bauen/installen
cd /pfad/zu/cadipy
export CADICAL=~/github/cadical
python -m pip install -U pip build
pip install .
# oder im Dev-Modus (editable):
pip install -e .
