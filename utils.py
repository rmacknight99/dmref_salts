import os, json, sys, shutil
import pandas as pd
import numpy as np
from morfeus import conformer, Dispersion, read_xyz, XTB, LocalForce, SASA, utils
from rdkit.Chem import AllChem as Chem

def get_dispersion(filename, desc_dict):

    elements, coordinates = read_xyz(filename)

    disp = Dispersion(elements, coordinates)
    disp.compute_coefficients()
    disp.compute_p_int()

    desc_dict["disp_area"] = disp.area
    desc_dict["disp_volume"] = disp.volume
    desc_dict["disp_p_int"] = disp.p_int
    desc_dict["disp_p_max"] = disp.p_max
    desc_dict["dos_p_min"] = disp.p_min

    return desc_dict

def get_SASA(filename, desc_dict):

    elements, coordinates = read_xyz(filename)
    sasa = SASA(elements, coordinates)
    desc_dict["sasa_area"] = sasa.area
    desc_dict["sasa_volume"] = sasa.volume
    
    return desc_dict

def get_XTB(filename, desc_dict):

    elements, coordinates = read_xyz(filename)
    xtb = XTB(elements, coordinates)

    desc_dict["ip"] = xtb.get_ip()
    desc_dict["ea"] = xtb.get_ea()                                                                                                               
    desc_dict["electrophilicity"] = xtb.get_global_descriptor("electrophilicity", corrected=True)
    desc_dict["nucleophilicity"] = xtb.get_global_descriptor("nucleophilicity", corrected=True)

    return desc_dict

def get_LFCs(filename, desc_dict):

    elements, coordinates = read_xyz(filename)
    LF = LocalForce(elements, coordinates)
    hess_path = f"hessian_files/{filename[:-4]}/hessian"
    LF.load_file(hess_path, "xtb", "hessian")
    LF.normal_mode_analysis()
    LF.detect_bonds()
    LF.compute_local()
    LF.compute_frequencies()
    LF.compute_compliance()

    desc_dict["local_force_constants"] = LF.local_force_constants
    desc_dict["local_frequencies"] = LF.local_frequencies

    return desc_dict, LF

def find_bond(elements, int_coords, search):

    for index, bond in enumerate(int_coords):
        idx_1 = bond.i - 1
        idx_2 = bond.j - 1
        key = elements[idx_1] + "_" + elements[idx_2]
        if key == search:
            return index

def resolve_LFCs(filename, desc_dict, LF, search):

    elements, coordinates = read_xyz(filename)
    index = find_bond(elements, LF.internal_coordinates, search)
    desc_dict["local_force_"+search] = desc_dict["local_force_constants"][index]
    desc_dict["local_frequencies_"+search] = desc_dict["local_frequencies"][index]
    desc_dict.pop("local_force_constants")
    desc_dict.pop("local_frequencies")

    return desc_dict

def get_ensemble_descriptors(salt_number, withh=True, dispersion=True, SASA=True, XTB=True, LFCs=None):

    descriptors = []
    HOME = os.getcwd()
    if withh:
        geom_dir = f"salt_{salt_number}/with_chlorine/CONFORMERS"
    else:
        geom_dir = f"salt_{salt_number}/without_chlorine/CONFORMERS"
    os.chdir(geom_dir)
    for i in os.listdir("."):
        print(f"\tworking on {geom_dir}")
        if ".xyz" in i:
            desc_dict = {}
            filename = i
            if dispersion:
                desc_dict = get_dispersion(filename, desc_dict)
            if SASA:
                desc_dict = get_SASA(filename, desc_dict)
            if XTB:
                desc_dict = get_XTB(filename, desc_dict)
            if LFCs is not None:
                desc_dict, LF = get_LFCs(filename, desc_dict)
                desc_dict = resolve_LFCs(filename, desc_dict, LF, LFCs)
            descriptors.append(desc_dict)
    os.chdir(HOME)
    for i, mol in enumerate(descriptors):
        for key, value in mol.items():
            descriptors[i][key] = round(value, 3)
        
    return descriptors

def gen_hess(salt_number, CORES, withh=True):
    
    HOME = os.getcwd()

    if withh:
        geom_dir = f"salt_{salt_number}/with_chlorine/CONFORMERS"
        chrg, uhf = 0, 0
    else:
        geom_dir = f"salt_{salt_number}/without_chlorine/CONFORMERS"
        chrg, uhf = 1, 0

    os.chdir(geom_dir)
    os.system("mkdir -p hessian_files/") # make hessian files
    ROOT = os.getcwd()
    for xyz_file in os.listdir("."): # iterate over xyz files
        if ".xyz" in xyz_file:
            name = xyz_file[:-4]
            if not os.path.exists(f"hessian_files/{name}/hessian"):
                os.system(f"mkdir -p hessian_files/{name}/")
                os.system(f"cp {xyz_file} hessian_files/{name}/")
                os.chdir(f"hessian_files/{name}/")
                os.system(f"xtb {xyz_file} --hess -c {chrg} -u {uhf} -P {CORES} >/dev/null 2>&1")
                os.chdir(ROOT)
    os.chdir(HOME)

def dump_conformers(ce, salt_number, withh=True):

    if withh:
        os.system(f"mkdir -p salt_{salt_number}/with_chlorine/CONFORMERS")
        ce.write_xyz(f"salt_{salt_number}/with_chlorine/CONFORMERS/conformer", separate=True)
    else:
        os.system(f"mkdir -p salt_{salt_number}/without_chlorine/CONFORMERS")
        ce.write_xyz(f"salt_{salt_number}/without_chlorine/CONFORMERS/conformer", separate=True)
        
def trim_conformers(ce, rmsd_thresh=0.35, energy_thresh=3.0):

    ce.prune_energy(threshold=energy_thresh)
    ce.prune_rmsd(method="obrms-batch", thres=rmsd_thresh)
    ce.sort()

    return ce

def make_best_ensemble(salt_number):

    with_ce, with_dict = load_ensemble(f"salt_{salt_number}/"+"with_chlorine/CREST/")
    without_ce, without_dict = load_ensemble(f"salt_{salt_number}/"+"without_chlorine/CREST/")
    elements = utils.convert_elements(without_dict["best_elements"], output="symbols")

    if list(with_dict["best_elements"]) == list(without_dict["best_elements"]):
        best_ensemble = conformer.ConformerEnsemble(elements=elements)
        best_ensemble.add_conformers([with_dict["best_coords"]], None, None, None)
        best_ensemble.add_conformers([without_dict["best_coords"]], None, None, None)
    
    #print(f"\tMean RMSD between crest bests: {best_ensemble.get_rmsd(method='obrms-iter').sum()/(best_ensemble.n_conformers*(best_ensemble.n_conformers-1))}")
    
    return with_ce, without_ce, best_ensemble

def remove_counterion(ce_elements):

    unique, counts = np.unique(ce_elements, return_counts=True)
    n_chlorines = dict(zip(unique, counts))[17]

    if n_chlorines > 1:
        ii = np.where(ce_elements == 17)[0][-1]
        ce_elements = list(ce_elements)
        ce_elements.pop(ii)
    else:
        ce_elements = list(ce_elements)
        ce_elements.remove(17)

    return ce_elements

def load_ensemble(path):

    ce = conformer.ConformerEnsemble.from_crest(path)
    ce_best_elements = ce.elements
    if "with_" in path:
        ce_best_elements = remove_counterion(ce_best_elements)
        ce_best_coords = np.delete(ce.get_coordinates()[0], (5), axis=0)
    else:
        ce_best_coords =ce.get_coordinates()[0]
        ce_best_elements = list(ce_best_elements)

    ddict = {"ce": ce, "best_coords": ce_best_coords, "best_degen": None, "best_elements": ce_best_elements, "best_energy": None}
    
    return ce, ddict

def run_crest_pipeline(input_file, cores):

    smiles = sorted([string + ".[Cl-]" for string in pd.read_csv(input_file, names=["x"])["x"].tolist()])
    for index, smiles_string in enumerate(smiles):
        print(f"-----On salt #{index}-----")
        with_chlorine = smiles_string
        without_chlorine = smiles_string.split(".")[0]
        print(with_chlorine)
        pipeline(with_chlorine, f"salt_{index}/with_chlorine", cores, 0, 1)
        pipeline(without_chlorine, f"salt_{index}/without_chlorine", cores, 1, 1)

def get_constraints(ID):

    os.system(f"obabel -ixyz {ID}/init.xyz -omol -O {ID}/init.mol")
    mol = Chem.MolFromMolFile(f"{ID}/init.mol", sanitize=False)
    os.remove(f"{ID}/init.mol")
    constrain_indices = []
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for index, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == "N":
            constrain_indices.append(index)
            bonded_atoms = [sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])[1] for b in atom.GetBonds()]
            bonded_atoms = [a for a in bonded_atoms if symbols[a] == "H"]
            constrain_indices += bonded_atoms
        elif atom.GetSymbol() == "Cl":
            constrain_indices.append(index)

    return constrain_indices

def write_xtb_inp(constraints):

    with open("xtb.inp", "w") as xtb_inp:
        xtb_inp.write("$fix\n")
        xtb_inp.write(f"\telements: N,H,O,Cl\n")
        xtb_inp.write("$end\n")

def check_xTB(ID):

    os.system("mkdir -p xTB_errors")
    with open(f"{ID}/XTB/xtb.out", "r") as f:
        lastline = f.readlines()[-1]
        lastline.replace("\n", "")
    if "#" in lastline:
        try:
            shutil.move(f"{ID}", "xTB_errors")
            print(f"\tmoved {ID} folders to xTB_errors", flush=True)
        except:
            pass

def pipeline(SMILES, ID, cores, charge, multiplicity=1):

    ROOT = os.getcwd()
    CREST_CORES = cores
    XTB_CORES = min(cores, 16)

    # Generate initial XYZ file from SMILES with openbabel
    if not os.path.exists(f"{ID}/init.xyz"):
        print(f"-----Generating Initial XYZ Structure for {SMILES}-----\n", flush=True)
        os.system(f"mkdir -p {ID}")
        os.system(f"obabel -:'{SMILES}' --addhs --gen3d -O {ID}/init.xyz > /dev/null 2>&1")

    # Record the SMILES
    os.system(f"obabel -ixyz {ID}/init.xyz -osmi -O {ID}/smiles.txt > /dev/null 2>&1")

    # Get the constraints
    constraints = get_constraints(ID)

    # Initial geometry optimization with xTB
    if not os.path.exists(f"{ID}/XTB/"):
        print(f"-----Optimizing Initial XYZ Structure with xTB-----", flush=True)
        try:
            os.chdir(ID)
            uhf = multiplicity - 1
            print(f"\tSMILES: {SMILES}\n\tCHARGE: {charge}\n\tUHF: {uhf}\n", flush=True)
            cmd = f"xtb init.xyz --opt -c {charge} -u {uhf} -P {XTB_CORES} --input xtb.inp > xtb.out"
            os.system("mkdir -p XTB")
            shutil.copy("init.xyz", "XTB/init.xyz")
            os.chdir("XTB")
            # Write constraint input
            write_xtb_inp(constraints)
            os.system(cmd)
            os.chdir(ROOT) # always return to root upon completion
        except:
            os.chdir(ROOT) # always return to root upond completion  
    
    if not os.path.exists(f"{ID}/XTB/xtbopt.xyz"):
        print(f"-----RERUNNING Initial Optimization at GFNFF level of theory-----", flush=True)
        try:
            os.chdir(ID)
            uhf = multiplicity - 1
            print(f"\tSMILES: {SMILES}\n\tCHARGE: {charge}\n\tUHF: {uhf}\n", flush=True)
            cmd = f"xtb init.xyz --opt -c {charge} -u {uhf} -P {XTB_CORES} --gfnff --input xtb.inp > xtb.out"
            os.system("mkdir -p XTB")
            shutil.copy("init.xyz", "XTB/init.xyz")
            os.chdir("XTB")
            os.system(cmd)
            os.chdir(ROOT) # always return to root upon completion
        except:
            os.chdir(ROOT) # always return to root upond completion  
    
    # check for xTB errors and move
    check_xTB(ID)

    # CREST conformer generation with gfn2//gfnff
    if not os.path.exists(f"{ID}/CREST/"):
        print(f"-----CREST Conformer Generation-----\n", flush=True)
        try:
            os.chdir(ID)
            os.system("mkdir -p CREST")
            shutil.copy("init.xyz", "CREST/init.xyz")
            os.chdir("CREST")
            os.system(f"crest init.xyz --gfn2//gfnff --chrg {charge} --uhf {uhf} --cbonds -T {CREST_CORES} --noreftopo > crest.out")
            os.chdir(ROOT) # always return to root upond completion  
        except:
            os.chdir(ROOT) # always return to root upond completion

    # If CREST not converged use GNF2
    if os.path.exists(f"{ID}/CREST/NOT_CONVERGED"):
        print(f"-----RERUNNING CREST with GFN2-----", flush=True)
        try:
            os.chdir(ID)
            os.system("mkdir -p CREST")
            shutil.copy("init.xyz", "CREST/init.xyz")
            os.chdir("CREST")
            os.system(f"crest init.xyz --gfn2 --chrg {charge} --uhf {uhf} --cbonds -T {CREST_CORES} --quick --noreftopo > crest.out")
            os.chdir(ROOT) # always return to root upond completion
        except:
            os.chdir(ROOT) # always return to root upond completion
