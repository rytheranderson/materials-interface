import sys
import re
import numpy as np
import glob

from itertools import groupby
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from ase.visualize import view
from ase.io import write

import matplotlib.pyplot as plt

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
    for i in xrange(1, int(np.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i*i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor

class material_grid:

	def __init__(self, mat_dict):
		self.mat_dict = mat_dict

	def build_grid(self, projection_vector=[0,0,1], realign=False, align_vec='a', square_grids=False):

		atoms = Atoms()

		mat_dict = self.mat_dict
		materials = sorted([(len(mat_dict[m]), m) for m in mat_dict], key = lambda x: x[0])
		
		divisors = list(divisorGenerator(len(mat_dict)))
		median = np.median(divisors)
		dists = [(i, abs(divisors[i] - median)) for i in range(len(divisors))]
		dists.sort(key = lambda x: x[1])
		nrow, ncol = [divisors[i[0]] for i in dists[0:2]]

		x = np.array([1,0,0])
		y = np.array([0,1,0])
		z = np.array([0,0,1])
		basis = [x,y,z]

		projection_vector = np.asarray(map(float, projection_vector))
		grid_plane_dims = [dim for dim in range(len(basis)) if not np.array_equal(basis[dim], projection_vector)]
		grid_plane_vectors = [basis[dim] for dim in grid_plane_dims]

		max_0 = 0
		max_1 = 0
		
		for entry, mat in mat_dict.iteritems():

			lengths = map(np.linalg.norm, mat.get_cell())
			l0 = lengths[grid_plane_dims[0]]
			l1 = lengths[grid_plane_dims[1]]

			if l0 > max_0:
				max_0 = l0 + 2.0
			if l1 > max_1:
				max_1 = l1 + 2.0

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
				grid_point = coord_grid[i,j]
				vecs = mat.positions
				unit_cell = mat.get_cell().T
				elems = [a.symbol for a in mat]
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
		return atoms

	def write_grid(self, filename='grid.xyz'):

		types = ['xyz', 'png', 'pdb']
		ftype = filename.split('.')[1]
		
		if ftype not in types:
			print ftype, 'not supported for writing grids'
			return 

		grid = self.build_grid()
		write(filename, grid)

	def view_grid(self):

		grid = self.build_grid()
		view(grid)


### Testing
#from enumerate_surface_adsorption import adsorbate_surface
#from materials_project_query import mp_query
#from ase import Atom, Atoms
#from write_outputs import write_adsorption_configs
#
#q = mp_query('ghLai1BTnNsvWZPu')
#m = q.make_structures('Cu')[0]
#
#a = adsorbate_surface(m, (1,0,0), 4, 4)
#a.make_supercell((3,2,1))
#
#d = 1.1
#CO = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)])
#N= Atoms('H', positions=[(0, 0, 0)])
#a.add_adsorbate_atoms( [(N, 0)], loading='all', name='ontop')
##write_adsorption_configs(a.adsorbate_configuration_dict)
#
#mat = material_grid(a.adsorbate_configuration_dict)
#grid = mat.build_grid(square_grids=True)
#mat.view_grid()

