import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

from matplotlib import pyplot as plt
import itertools
import scipy
import re

from pymatgen import Structure, MPRester, Molecule
from pymatgen.analysis.adsorption import *
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atom, Atoms
from ase.build import make_supercell
from ase.spacegroup import *

def all_subsets(l):
	return itertools.chain(*map(lambda x: itertools.combinations(l, x), range(0, len(l)+1)))

def isfloat(value):
	try:
		float(value)
		return True
	except ValueError:
		return False

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

class adsorbate_surface():

	""" 
		Input is a pymatgen bulk structure, the desired surface plane, 
		the desired slab depth (in layers), and the vacuum size (also in layers).
	""" 

	def __init__(self, struct, plane, slab_depth, vacuum_size):

		mi = max(plane)
		slabs = generate_all_slabs(struct, 
								   min_slab_size=slab_depth, 
								   min_vacuum_size=vacuum_size, 
								   max_index=mi, 
								   in_unit_planes=True, 
								   center_slab=True)

		slab = [slab for slab in slabs if slab.miller_index==plane][0]
		slab = slab.get_orthogonal_c_slab()
		ase_atoms = AseAtomsAdaptor.get_atoms(slab)
	
		self.blank_slab_pym    = slab
		self.blank_slab_ase    = ase_atoms
		self.minimal_unit_cell = ase_atoms.get_cell().T
		self.duplications      = (1,1,1)
		self.plane             = plane

	def plot_ads_sites(self, filename='ads_sites.tiff', size=(3,3)):

		""" Makes a nice plot of the identified sites. """
		
		fig = plt.figure()
		plt.axis('off')
		fig.set_size_inches(size[0],size[1])
		ax = fig.add_subplot(111)
		plot_slab(self.blank_slab_pym, ax, adsorption_sites=True)
		plt.savefig(filename, dpi=300, bbox_inches='tight')

	def make_supercell(self, replicate):

		""" 
			Replicates the slab (ase version) according to replicate, 
			which is an iterable with the number of replications in 
			each direction. E.g. (2,2,1) would double the slab in 
			the a and b directions, and leave the c direction unchanged.
		"""

		r1,r2,r3 = replicate
		P = np.array([[r1,0,0],[0,r2,0],[0,0,r3]])
		self.blank_slab_ase = make_supercell(self.blank_slab_ase, P)
		self.duplications = replicate

	def add_adsorbate_atoms(self, adsorbate, loading='all', name='all', ):

		"""
			Places adsorbates according to loading. The ads parameter is
			a list of the adsorbates to be used. If loading is 'all' every
			configuration of all possible loadings will be made. Otherwise 
			loading should be a float between 0 and 1 indicating the desired
			fraction of a monolayer.
		"""

		asf = AdsorbateSiteFinder(self.blank_slab_pym)
		ads_sites = asf.find_adsorption_sites()

		duplications = self.duplications
		unit_cell = self.minimal_unit_cell
		repeat_unit_cell = self.blank_slab_ase.get_cell().T

		inv_ruc = np.linalg.inv(repeat_unit_cell)
		plane_string = ''.join(map(str, self.plane))

		duplications = [[j for j in range(i)] for i in duplications]
		translation_vecs = list(itertools.product(duplications[0], duplications[1], duplications[2]))
		ads_positions = ads_sites[name]
		all_combinations = [s for s in all_subsets(translation_vecs) if len(s) > 0]
		monolayer_loading = float(max(map(len, all_combinations)))
		
		if isfloat(loading):

			fractional_loadings = [abs(len(s)/monolayer_loading - loading) for s in all_combinations]
			closest = min(set(fractional_loadings))
			all_combinations = [all_combinations[i] for i in range(len(all_combinations)) if fractional_loadings[i] == closest]			

		ads_dict = {}
		atype_counter = 0

		for pos in ads_positions:
			atype_counter += 1

			for subset in all_combinations:

				loading = len(subset)
				fractional_loading = int((loading/monolayer_loading)*100)
				adsorbate_combinations = itertools.product(adsorbate, repeat=loading)

				for ads_comb in adsorbate_combinations:
					
					adsonly_positions = []
					slab_coords = [(sl.symbol, sl.position) for sl in self.blank_slab_ase]
					
					for s,a in zip(subset, ads_comb):

						ads, shift_ind = a

						elems = []
						positions = []
						for atom in ads:
							elems.append(atom.symbol)
							positions.append(atom.position)
						elems = np.asarray(elems)
						position = np.asarray(positions)

						trans = np.dot(unit_cell, np.asarray(s))
						shift_vec = pos - positions[shift_ind] + trans
						positions += shift_vec

						for e,c in zip(elems, positions):
							slab_coords.append((e, c))
							adsonly_positions.append(c)

					atoms = Atoms()
					for c in slab_coords:
						atoms.append(Atom(c[0],c[1]))
					
					formula = atoms.get_chemical_formula()
					atoms.set_cell(repeat_unit_cell)
					struct = AseAtomsAdaptor.get_structure(atoms)
					sga = SpacegroupAnalyzer(struct, symprec=0.1)
					sgs = sga.get_space_group_symbol()
					sgn = sga.get_space_group_number()

					adsonly_positions = np.asarray(adsonly_positions) - np.average(ads_positions, axis=0)

					dists = []
					for i in range(len(adsonly_positions)):
						icoord = np.dot(inv_ruc, adsonly_positions[i])
						for j in range(i + 1, len(adsonly_positions)):
							jcoord = np.dot(inv_ruc, adsonly_positions[j])
							fdist, sym = PBC3DF_sym(icoord,jcoord)
							dist = np.linalg.norm(np.dot(repeat_unit_cell, fdist))
							dists.append(dist)

							#print icoord, jcoord, dist, sgs

					dists.sort()
					dists = np.asarray([np.round(d, 1) for d in dists]) * 10
					fp1 = int(np.average(dists))
					fp2 = int(max(dists))
					fp3 = int(min(dists))
					fp = str(fp1) + '-' + str(fp2) + '-' + str(fp3)

					ads_dict[formula + '_' + plane_string + '_' + str(fractional_loading) + '_' + name + str(atype_counter) + '_' + sgs + '_' + fp] = atoms

		self.adsorbate_configuration_dict = ads_dict

#Testing
from materials_project_query import mp_query
from write_output_files import write_adsorption_configs

q = mp_query('ghLai1BTnNsvWZPu')
m = q.make_structures('Cu')[0]

a = adsorbate_surface(m, (1,0,0), 4, 4)
a.make_supercell((3,2,1))

d = 1.1
CO = Atoms('CO', positions=[(0, 0, 0), (0, 0, d)])
N= Atoms('H', positions=[(0, 0, 0)])
a.add_adsorbate_atoms( [(N, 0)], loading=0.5, name='ontop')
write_adsorption_configs(a.adsorbate_configuration_dict)

