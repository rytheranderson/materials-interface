#!/usr/bin/env python

import sys
import re
import numpy as np
import glob

from itertools import groupby
from numpy import cross, eye, dot
from scipy.linalg import expm, norm

import matplotlib.pyplot as plt

PT = ['H' , 'He', 'Li', 'Be', 'B' , 'C' , 'N' , 'O' , 'F' , 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
	  'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
	  'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe',
	  'Cs', 'Ba', 'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 
	  'Ra', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Ac', 'Th', 
	  'Pa', 'U' , 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'FG', 'X' ]

def nn(string):
	return re.sub('[^a-zA-Z]','', string)

def nl(string):
	return re.sub('[^0-9]','', string)

def isfloat(value):
	"""
		determines if a value is a float
	"""
	try:
		float(value)
		return True
	except ValueError:
		return False

def iscoord(line):
	"""
		identifies coordinates in CIFs
	"""
	if nn(line[0]) in PT and line[1] in PT and False not in map(isfloat,line[2:5]):
		return True
	else:
		return False
	
def isbond(line):
	"""
		identifies bonding in cifs
	"""
	if nn(line[0]) in PT and nn(line[1]) in PT and isfloat(line[2]) and True not in map(isfloat,line[3:]):
		return True
	else:
		return False

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

	def cif_grid(self, key, key_header=True, projection_vector=[0,0,1], realign=False, align_vec='a', square_grids=True):

		cif_dict = self.cif_dict

		x = np.array([1.0,0.0,0.0])
		y = np.array([0.0,1.0,0.0])
		z = np.array([0.0,0.0,1.0])

		basis = [x,y,z]
		projection_vector = np.asarray(map(float, projection_vector))
		grid_plane_dims = [dim for dim in range(len(basis)) if not np.array_equal(basis[dim], projection_vector)]
		grid_plane_vectors = [basis[dim] for dim in grid_plane_dims]

		max_0 = 0
		max_1 = 0
		for cif in cif_dict:
			lengths = cif_dict[cif].lengths
			l0 = lengths[grid_plane_dims[0]]
			l1 = lengths[grid_plane_dims[1]]

			if l0 > max_0:
				max_0 = l0 + 5.0
			if l1 > max_1:
				max_1 = l1 + 5.0

		if square_grids:
			max_0 = max(max_0, max_1)
			max_1 = max(max_0, max_1)
			
		org = np.genfromtxt(key, delimiter='', dtype=None)

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

		cif_grid = []
		count = 0
		for v in sorted(row_values):
			row = []
			for i in row_org_dict[v]:
				row.append((i, column_org_dict[i]))
			cif_grid.append(sorted(row, key = lambda x: x[1]))
		cif_grid = np.asarray(cif_grid)

		nrow = len(row_values)
		ncol = len(column_values)
		
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
		mof_count = 0

		color_grid = np.zeros((nrow,ncol))

		print cif_grid.shape, nrow, ncol

		for i in range(nrow):
			for j in range(ncol):

				print i,j

				mof_count +=1

				mof = cif_grid[i,j][0]
				grid_point = coord_grid[i,j]
				coords = cif_dict[mof].cart_coords
				unit_cell = cif_dict[mof].unit_cell.T
				color = color_dict[mof]

				color_grid[i,j] += float(color)

				elems = [nn(l[0]) for l in coords]
				vecs  = np.array([l[1] for l in coords])

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
					all_atoms.append((e,v,mof_count,mof,float(color)))
		
		self.mof_grid = all_atoms
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

cif_list = glob.glob('*.cif')
foo = cif_group(cif_list)
foo.cif_grid('key.txt', realign=False, align_vec='b')
foo.save_color_grid(cmap='viridis', alpha=1.0)
foo.write_pdb()











