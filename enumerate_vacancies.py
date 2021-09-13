import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

import itertools
import numpy as np

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atom, Atoms
from ase.build import make_supercell

def PBC3DF_sym(vec1, vec2):

    dX,dY,dZ = vec1 - vec2
            
    if dX > 0.5:
        s1 = 1
        ndX = dX - 1.0
    elif dX < -0.5:
        s1 = -1
        ndX = dX + 1.0
    else:
        s1 = 0
        ndX = dX
                
    if dY > 0.5:
        s2 = 1
        ndY = dY - 1.0
    elif dY < -0.5:
        s2 = -1
        ndY = dY + 1.0
    else:
        s2 = 0
        ndY = dY
    
    if dZ > 0.5:
        s3 = 1
        ndZ = dZ - 1.0
    elif dZ < -0.5:
        s3 = -1
        ndZ = dZ + 1.0
    else:
        s3 = 0
        ndZ = dZ

    return np.array([ndX,ndY,ndZ]), np.array([s1,s2,s3])

def redundancy_check(blank_coords, adsonly_positions, fp_dict, repeat_unit_cell, symmetry_tol):

    inv_ruc = np.linalg.inv(repeat_unit_cell)

    atoms = Atoms()
    for c in blank_coords:
        atoms.append(Atom(c[0],c[1]))
    
    formula = atoms.get_chemical_formula()
    atoms.set_cell(repeat_unit_cell.T)
    atoms.pbc = True
    struct = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(struct, symprec=symmetry_tol)
    sgs = sga.get_space_group_symbol()
    sgn = sga.get_space_group_number()

    adsonly_positions = np.asarray(adsonly_positions)

    dists = []
    for i in range(len(adsonly_positions)):
        icoord = np.dot(inv_ruc, adsonly_positions[i])
        for j in range(i + 1, len(adsonly_positions)):
            jcoord = np.dot(inv_ruc, adsonly_positions[j])
            fdist, sym = PBC3DF_sym(icoord,jcoord)
            dist = np.linalg.norm(np.dot(repeat_unit_cell, fdist))
            dists.append(dist)
    dists.sort() 

    index = 0
    advance = True
    for ind in range(len(fp_dict[sgn])):
        fp = fp_dict[sgn][ind]
        if len(fp) == len(dists):
            if np.allclose(fp, dists):
                index = ind
                advance = False
                break
    index = str(index)

    return advance, index, sgs, sgn, dists, atoms, formula

class vacancy_structure_generator(object):

    def __init__(self, struct):

        ase_atoms = AseAtomsAdaptor.get_atoms(struct)

        self.struct = struct
        self.ase_atoms = ase_atoms
        self.minimal_unit_cell = ase_atoms.get_cell().T

    def make_supercell(self, replicate):

        r1,r2,r3 = replicate
        P = np.array([[r1,0,0],[0,r2,0],[0,0,r3]])
        self.ase_atoms = make_supercell(self.ase_atoms, P)
        self.duplications = replicate

    def make_vacancies(self, nvacancies, elements='all', symmetry_tol=0.01):

        vacancy_dict = {}
        ase_atoms = self.ase_atoms
        repeat_unit_cell = ase_atoms.get_cell().T

        atoms_to_remove = []
        for e in ase_atoms:
            if elements != 'all':
                if e.symbol in elements:
                    atoms_to_remove.append(e.index)
            else:
                atoms_to_remove.append(e.index)

        all_combinations = list(itertools.combinations(atoms_to_remove, nvacancies))
        fp_dict   = dict((k,[]) for k in range(1,231))
        sg_counts = dict((k,0 ) for k in range(1,231))
        
        for comb in all_combinations:

            remove_indices = [i for i in comb]
            remove_coords  = [ase_atoms[i].position for i in comb]
            remaining_coords = [(i.symbol, i.position) for i in ase_atoms if i.index not in remove_indices]

            advance, index, sgs, sgn, dists, atoms, formula = redundancy_check(remaining_coords, remove_coords, fp_dict, repeat_unit_cell, symmetry_tol)

            if advance:
                sg_counts[sgn] += 1
                fp_dict[sgn].append(dists)

            vacancy_dict[formula + '_' + str(nvacancies) + '_' + str(sgn) + '_' + index] = atoms

        self.vacancy_dict = vacancy_dict

#from read_inputs import mp_query, file_read
#from write_outputs import write_adsorption_configs

#m = file_read('NiO.cif')
#m, a = m.read_cif()
#vsg = vacancy_structure_generator(m)
#vsg.make_supercell((2,1,1))
#vsg.make_vacancies(0, elements=['O'])
#write_adsorption_configs(vsg.vacancy_dict, filetype='vasp')
