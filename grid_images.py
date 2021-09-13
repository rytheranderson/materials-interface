import numpy as np
from itertools import combinations
from numpy import cross, eye
from scipy.linalg import expm, norm
from ase.visualize import view
from ase.io import write
from ase import Atom, Atoms

metals = ('Pd', 'V', 'Cu', 'Cr', 'Fe', 'Au', 'Ag')

def R(axis, theta):
    """
        returns a rotation matrix that rotates a vector around axis by angle theta
    """
    return expm(cross(eye(3), axis/norm(axis)*theta))

def M(vec1, vec2):
    """
        returns a rotation matrix that rotates vec1 onto vec2
    """
    ax = np.cross(vec1, vec2)
    if np.any(ax):
        ax_norm = ax/np.linalg.norm(np.cross(vec1, vec2))
        foo = np.dot(vec1, vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
        ang = np.arccos(foo)
        return R(ax_norm, ang)
    else:
        return np.asarray([[1,0,0],[0,1,0],[0,0,1]])

def PBC3DF_sym(vec1, vec2):
    """
        calculates distance vector between vec1 and vec2 accoring to the MIC, 
        and returns translation vector required to transform vec2 across boundaries
        according to the MIC (sym). These opertations are done in fractional coordinates
        to make things easier. Any computation in fractional space can easily be converted 
        to Cartesian space using the unit_cell matrix defined in the __init__ function
        of the cif_convert class defined below.
    """

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

def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

class material_grid(object):

    def __init__(self, mat_dict):
        self.mat_dict = mat_dict

    def build_grid(self, projection_vector=[0,0,1], realign=False, align_vec='a', square_grids=False, duplicate_border_atoms=True, duplicate_dims=(0,1), duplicate_elems=metals, sep_dist=5.0):

        atoms = Atoms()

        mat_dict = self.mat_dict
        materials = sorted([(len(mat_dict[m]), m) for m in mat_dict], key = lambda x: x[0])
        
        divisors = list(divisorGenerator(len(mat_dict)))
        median = np.median(divisors)
        dists = [(i, abs(divisors[i] - median)) for i in range(len(divisors))]
        dists.sort(key = lambda x: x[1])
        nrow, ncol = [int(divisors[i[0]]) for i in dists[0:2]]

        #nrow = 1
        #ncol = len(materials)

        x = np.array([1,0,0])
        y = np.array([0,1,0])
        z = np.array([0,0,1])
        basis = [x,y,z]

        projection_vector = np.asarray([float(v) for v in projection_vector])
        grid_plane_dims = [dim for dim in range(len(basis)) if not np.array_equal(basis[dim], projection_vector)]

        max_0 = 0
        max_1 = 0
        
        for entry, mat in mat_dict.items():

            lengths = list(map(np.linalg.norm, mat.get_cell()))
            l0 = lengths[grid_plane_dims[0]]
            l1 = lengths[grid_plane_dims[1]]

            if l0 > max_0:
                max_0 = l0 + sep_dist
            if l1 > max_1:
                max_1 = l1 + sep_dist

        if square_grids:
            max_0 = max(max_0, max_1)
            max_1 = max(max_0, max_1)
        
        val  = max_0/2.0
        mid0 = [val]
        for i in range(nrow - 1):
            val += max_0
            mid0.append(val)

        val  = max_1/2.0
        mid1 = [val]
        for i in range(ncol - 1):
            val += max_1
            mid1.append(val)

        coord_grid = []
        for i in mid0:
            row = []
            for j in mid1:
                row.append([i,j])
            coord_grid.append(row)
        coord_grid = np.asarray(coord_grid)

        mat_count = 0
        for i in range(nrow):
            for j in range(ncol):

                mat_name = materials[mat_count][1]
                mat = mat_dict[mat_name]
                elems = [a.symbol for a in mat]
                grid_point = coord_grid[i,j]
                
                vecs = mat.positions
                unit_cell = mat.get_cell().T
                fvecs = np.dot(np.linalg.inv(unit_cell), vecs.T).T

                if duplicate_border_atoms:

                    basis = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
                    new_vecs = []

                    for elem,fcoord in zip(elems,fvecs):

                        if elem in duplicate_elems:

                            zero_threshold_indices = fcoord < 1e-4
                            fcoord[zero_threshold_indices] = 0.0
                            one_threshold_indices = abs(fcoord - 1.0) < 1e-4
                            fcoord[one_threshold_indices]  = 1.0
                            dup_vecs = [basis[dim] for dim in (0,1,2) if fcoord[dim] == 0.0]
                            combs = list(combinations(dup_vecs, 3)) + list(combinations(dup_vecs, 2))
                            
                            for comb in combs:
                                compound = np.array([0.0,0.0,0.0])
                                for vec in comb:
                                    compound += vec
                                dup_vecs.append(compound)
                                
                            for vec in dup_vecs:
                                new_coord = [np.round(n, 6) for n in np.dot(unit_cell, fcoord + vec)]
                                new_vecs.append(new_coord)
                                elems.append(elem)
                    
                    new_vecs = np.asarray(new_vecs)
                    vecs = np.r_[vecs, new_vecs] 

                com = np.average(vecs, axis=0)
                plane_point = np.array([0.0,0.0,0.0])
                plane_point[grid_plane_dims[0]] = grid_point[0]
                plane_point[grid_plane_dims[1]] = grid_point[1]

                if realign:
                    if align_vec == 'a':
                        uc_vec = unit_cell[0]
                    elif align_vec == 'b':
                        uc_vec = unit_cell[1]
                    elif align_vec == 'c':
                        uc_vec = unit_cell[2]
                    else:
                        uc_vec = unit_cell[0]

                    rot = M(np.asarray(uc_vec), projection_vector)
                    vecs = np.dot(rot, (vecs - com).T).T

                translate = plane_point - com
                vecs = vecs + translate

                for e,v in zip(elems, vecs):
                    atoms.append(Atom(e,v))

                mat_count +=1

        self.grid = atoms

    def write_grid(self, filename='grid.xyz'):

        types = ['xyz', 'png', 'pdb']
        ftype = filename.split('.')[1]
        
        if ftype not in types:
            print(ftype, 'not supported for writing grids')
            return 

        write(filename, self.grid)

    def view_grid(self):

        view(self.grid)

### Testing
#from enumerate_adsorption import surface_adsorption_generator, bulk_adsorption_generator
#from ase import Atom, Atoms
#from write_outputs import write_adsorption_configs
#from read_inputs import mp_query, file_read
#from enumerate_vacancies import vacancy_structure_generator

#a = surface_adsorption_generator(m, (1,1,1), 2, 2)
#a.make_supercell((3,3,1))
#a.enumerate_ads_chains([(C,0)], 1.615, 6)
#mat = material_grid(a.path_configuration_dict)
#grid = mat.build_grid(square_grids=True)
#mat.view_grid()
#mat.write_grid('Cchains.xyz')
#
#a = bulk_adsorption_generator(m)
#a.Voronoi_tessalate()
#a.enumerate_ads_config([(C,0)], 12)
#mat = material_grid(a.adsorbate_configuration_dict)
#grid = mat.build_grid(square_grids=True)
#mat.view_grid()
#mat.write_grid('FCC_voronoi.xyz')
#
#a = bulk_adsorption_generator(m)
#a.Voronoi_tessalate()
#a.enumerate_ads_config([(N,0)], 12)
#write_adsorption_configs(a.adsorbate_configuration_dict, filetype='cif')
#
#m = file_read('PdO.cif')
#m, a = m.read_cif()
#vsg = vacancy_structure_generator(m)
#vsg.make_supercell((2,2,1))
#vsg.make_vacancies(4, elements=['O'])
#mat = material_grid(vsg.vacancy_dict)
#write_adsorption_configs(vsg.vacancy_dict, filetype='cif')
#grid = mat.build_grid(square_grids=True, duplicate_border_atoms=False, sep_dist=1.5)
#mat.view_grid()
#mat.write_grid('4vacancies.xyz')
#write_adsorption_configs(vsg.vacancy_dict, filetype='vasp')
#
#a = bulk_adsorption_generator(m)
#a.Voronoi_tessalate()
#a.enumerate_ads_config([(H,0)], 12)
#mat = material_grid(a.adsorbate_configuration_dict)
#grid = mat.build_grid(square_grids=True)
##mat.view_grid()
#mat.write_grid('12H.png')




