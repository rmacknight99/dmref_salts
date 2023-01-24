from utils import *

conf_dict = {}
total = 0

for i in list(range(58)):
    print(f"----- Salt #{i} -----")
    with_ce, without_ce, best_ce = make_best_ensemble(i)
    ce = trim_conformers(with_ce, rmsd_thresh=0.25, energy_thresh=3.0)
    conf_dict[i] = (ce, ce.n_conformers)
    total += ce.n_conformers

    dump_conformers(with_ce, i, withh=True)
    dump_conformers(without_ce, i, withh=False)
    gen_hess(i, 8, withh=True)
    gen_hess(i, 8, withh=False)

print(total // 58)
