import sys
import re
import numpy as np
import glob

from itertools import groupby
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from ase.visualize import view

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

class material_grid:

	def __init__(self, mat_dict):
		self.mat_dict = mat_dict

	def make_key(self, build='default'):

		mat_dict = self.mat_dict

		if build == 'file':
			key = np.genfromtxt(key, delimiter='', dtype=None)
		
		elif build == 'default':
			
			name = ['name']
			rows = ['rows']
			cols = ['cols']
			colo = ['colo']

			nentries = len(mat_dict)

			count = 0
			for m in mat_dict:
				count += 1
				entry = mat_dict[m]
				name.append(m)
				rows.append(m)
				colo.append(1.0)
				cols.append(count % 2)

			key = np.c_[name, rows, cols, colo]

		return key

	def build_grid(self, key_arg='default', key_header=True, projection_vector=[0,0,1], realign=False, align_vec='a', square_grids=True):

		atoms = Atoms()

		mat_dict = self.mat_dict
		org = self.make_key(key_arg)

		x = np.array([1.0,0.0,0.0])
		y = np.array([0.0,1.0,0.0])
		z = np.array([0.0,0.0,1.0])

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
				max_0 = l0 + 3.0
			if l1 > max_1:
				max_1 = l1 + 3.0

		if square_grids:
			max_0 = max(max_0, max_1)
			max_1 = max(max_0, max_1)

		if key_header:
			org = org[1:]

		row_org    = np.c_[org[:,0], org[:, 1]]
		column_org = np.c_[org[:,0], org[:, 2]]
		color_org  = np.c_[org[:,0], org[:, 3]]

		colors = np.array(map(float, color_org[:,1]))
		colors = colors/max(colors)
		color_org = np.c_[color_org[:,0],colors]

		color_dict = {}
		for l in color_org:
			color_dict[l[0]] = l[1]

		row_values = set(row_org[:,1])
		column_values = set(column_org[:,1])

		row_org_dict = dict((k,[]) for k in row_values)
		column_org_dict = {}

		for l in row_org:
			row_org_dict[l[1]].append(l[0])

		for l in column_org:
			column_org_dict[l[0]] = l[1]

		mat_grid = []
		for v in sorted(row_values):
			row = []
			for i in row_org_dict[v]:
				row.append((i, column_org_dict[i]))
			mat_grid.append(sorted(row, key = lambda x: x[1]))
		mat_grid = np.asarray(mat_grid)
		
		nrow, ncol, nentry = mat_grid.shape
		
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

		all_atoms = []
		mat_count = 0

		color_grid = np.zeros((nrow,ncol))

		for i in range(nrow):
			for j in range(ncol):

				mat_count +=1	
				mat_name = mat_grid[i,j][0]
				mat = mat_dict[mat_name]
				grid_point = coord_grid[i,j]
				vecs = mat.positions
				unit_cell = mat.get_cell().T
				color = color_dict[mat_name]

				color_grid[i,j] += float(color)

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
		
		self.grid = atoms
		self.color_grid = color_grid

	def write_pdb(self, print_to_screen=True):

		all_atoms = self.mof_grid

		count = 0
		#print 'COMPND mof_grid'
		#print 'COMPND created by mof_grid.py by Ryther Anderson'
		for l in all_atoms:
			count += 1
			line = ['HETATM', count, l[0], l[1][0], l[1][1], l[1][2], '1.00', l[4], l[0]]
			print '{:6} {:>5} {:>7} {:>16.3f} {:7.2f} {:7.2f} {:>5} {:>5.3f} {:>12}'.format(*line)

	def save_color_grid(self, cmap='hot', alpha=0.5):

		color_grid = self.color_grid
		w,h = plt.figaspect(color_grid)
		fig, ax = plt.subplots(figsize=(w,h), dpi=300)
		fig.subplots_adjust(0,0,1,1)

		plt.axis('off')

		ax.imshow(color_grid, cmap=cmap, alpha=alpha)
		
		fig.savefig('color_grid.tiff')

		cmap = plt.cm.get_cmap(cmap)
		colors = cmap(np.arange(cmap.N))

		fig, ax = plt.subplots(figsize=(w,h), dpi=300)
		fig.subplots_adjust(0,0,1,1)
		ax.imshow(color_grid, origin='lower', cmap=cmap, alpha=alpha)

		plt.axis('off')
		ax.imshow([colors], extent=[0, 10, 0, 1])

		fig.savefig('color_bar.tiff')

#Testing
from enumerate_surface_adsorption import adsorbate_surface
from materials_project_query import mp_query
from ase import Atom, Atoms
from write_output_files import write_adsorption_configs

q = mp_query('ghLai1BTnNsvWZPu')
m = q.make_structures('Cu')[0]

a = adsorbate_surface(m, (1,0,0), 4, 4)
a.make_supercell((3,3,1))

d = 1.1
CO = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)])
N= Atoms('H', positions=[(0, 0, 0)])
a.add_adsorbate_atoms( [(N, 0)], loading=0.5, name='ontop')
write_adsorption_configs(a.adsorbate_configuration_dict)

mat = material_grid(a.adsorbate_configuration_dict)
mat.build_grid()
#view(mat.grid)
#mat.make_key(build='default')

