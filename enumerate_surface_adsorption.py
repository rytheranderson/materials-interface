import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

from matplotlib import pyplot as plt
import networkx as nx
import itertools
from scipy.spatial import Voronoi
import re

from pymatgen import Structure
from pymatgen.analysis.adsorption import *
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atom, Atoms
from ase.build import make_supercell
from ase.constraints import FixAtoms
from ase.io import write

symmetry_order_dict = {24:'tetra', 48:'fcc_octa', 8:'bcc'}

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

class surface_adsorption_generator():

	""" 
		Input is a pymatgen bulk structure, the desired surface plane, 
		the desired slab depth (in layers), and the vacuum size (also in layers).
	""" 

	def __init__(self, struct, plane, slab_depth, vacuum_size, max_normal_search=5, symmetrize=True):

		mi = max(plane)
		slabs = generate_all_slabs(struct, 
								   min_slab_size=slab_depth, 
								   min_vacuum_size=vacuum_size, 
								   max_index=mi, 
								   in_unit_planes=True, 
								   center_slab=True,
								   symmetrize=symmetrize,
								   max_normal_search=max_normal_search)

		slab = [slab for slab in slabs if slab.miller_index==plane][0]

		ase_atoms = AseAtomsAdaptor.get_atoms(slab)
	
		self.blank_slab_pym    = slab
		self.blank_slab_ase    = ase_atoms
		self.minimal_unit_cell = ase_atoms.get_cell().T
		self.duplications      = (1,1,1)
		self.plane             = plane

	def write_blank_slab(self, filetype='cif'):

		slab = self.blank_slab_ase
		formula = slab.get_chemical_formula()
		plane_string = ''.join(map(str, self.plane))

		write(formula + '_' + plane_string + '.' + filetype, slab, format=filetype)

	def plot_ads_sites(self, filename='ads_sites.png', size=(3,3)):

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

	def enumerate_ads_config(self, adsorbate, loading='all', name='all', interaction_dist=2.0, symmetry_tol=0.01):

		"""
			Places adsorbates according to loading. The ads parameter is
			a list of the adsorbates to be used. If loading is 'all' every
			configuration of all possible loadings will be made. Otherwise 
			loading should be a float between 0 and 1 indicating the desired
			fraction of a monolayer.
		"""

		asf = AdsorbateSiteFinder(self.blank_slab_pym)
		ads_sites = asf.find_adsorption_sites(distance=interaction_dist)

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

		site_list = []
		if name == 'all':
			for stype in ads_sites:
				if stype != 'all':
					site_list.extend([s for s in ads_sites[stype]])
		else:
			site_list.extend([s for s in ads_sites[name]])
		
		if isfloat(loading):

			fractional_loadings = [abs(len(s)/monolayer_loading - loading) for s in all_combinations]
			closest = min(set(fractional_loadings))
			all_combinations = [all_combinations[i] for i in range(len(all_combinations)) if fractional_loadings[i] == closest]			

		ads_dict = {}
		atype_counter = 0

		fingerprints = dict((len(s), dict((k,[]) for k in range(1,231))) for s in all_combinations)
		sg_counts    = dict((k,0) for k in range(1,231))

		for pos in site_list:
			atype_counter += 1

			for subset in all_combinations:

				loading = len(subset)
				fp_dict = fingerprints[loading]

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
						positions = np.asarray(positions)
						trans = np.dot(unit_cell, np.asarray(s))
						positions -= positions[shift_ind]
						positions += pos
						positions += trans

						for e,c in zip(elems, positions):
							slab_coords.append((e, c))
							adsonly_positions.append(c)

					advance, index, sgs, sgn, dists, atoms, formula = redundancy_check(slab_coords, adsonly_positions, fp_dict, repeat_unit_cell, symmetry_tol)

					if advance:
						sg_counts[sgn] += 1
						fp_dict[sgn].append(dists)
						
					ads_dict[formula + '_' + plane_string + '_' + str(fractional_loading) + '_' + name + str(atype_counter) + '_' + sgs + '_' + index] = atoms

		self.adsorbate_configuration_dict = ads_dict

	def enumerate_ads_chains(self, adsorbate, bond_length, path_length, percent_error=10, include_cycles=True, mode='exact', symmetry_tol=0.01):
		
		bond_length = float(bond_length)
		asf = AdsorbateSiteFinder(self.blank_slab_pym)
		ads_sites = asf.find_adsorption_sites()

		site_list = []
		for stype in ads_sites:
			if stype != 'all':
				site_list.extend([s for s in ads_sites[stype]])

		duplications = self.duplications
		unit_cell = self.minimal_unit_cell
		repeat_unit_cell = self.blank_slab_ase.get_cell().T
		inv_ruc =  np.linalg.inv(repeat_unit_cell)
		plane_string = ''.join(map(str, self.plane))

		dists = []
		for pair in itertools.combinations(site_list, 2):
			s0, s1 = pair
			s0 = np.dot(np.linalg.inv(unit_cell), s0)
			s1 = np.dot(np.linalg.inv(unit_cell), s1)
			fdist, sym = PBC3DF_sym(s0, s1)
			dist = np.round(np.linalg.norm(np.dot(unit_cell, fdist)), 3)
			dists.append((dist, pair))

		dists = [d for d in dists if abs(d[0] - bond_length)/bond_length * 100 < percent_error]

		duplications = [[j for j in range(i)] for i in duplications]
		translation_vecs = list(itertools.product(duplications[0], duplications[1], duplications[2]))
		
		lattice_count = 0

		path_dict = {}
		ptype_counter = 0
		
		for d in dists:
			ptype_counter += 1

			fingerprints = dict((s, dict((k,[]) for k in range(1,231))) for s in range(path_length + 1))
			sg_counts    = dict((k,0) for k in range(1,231))
			lattice = []

			for s in d[1]:
				for trans in translation_vecs:
					trans = np.asarray(trans)
					cart_trans = np.dot(unit_cell, trans)
					lattice.append(s + cart_trans)

			G = nx.Graph()
			for i in range(len(lattice)):
				G.add_node(i, coords=lattice[i])

			for i in range(len(lattice)):
				s0 = lattice[i]
				for j in range(i + 1, len(lattice)):
					s1 = lattice[j]
					dist = np.linalg.norm(s0 - s1)
					if np.round(dist,3) == d[0]:
						G.add_edge(i, j, length=dist)

			def neighborhood(G, node, n):
				path_lengths = nx.single_source_dijkstra_path_length(G, node)
				return [node for node, length in path_lengths.iteritems() if length == n]

			all_paths = []
			all_cycles = []
			used = []
			for i in G.nodes():

				used.append(i)
				nborhood = [neighborhood(G, i, n) for n in range(path_length + 1)]
				nborhood = [nbor for nbors in nborhood for nbor in nbors]

				for j in nborhood:
					if j not in used:
						paths = list(nx.all_simple_paths(G, source=i, target=j, cutoff=path_length))
						for p in paths:
							if mode == 'leq':
								if len(p) <= path_length:
									all_paths.append(p)
							elif mode == 'exact':
								if len(p) == path_length:
									all_paths.append(p)

			if include_cycles:
				
				G = G.to_directed()

				if mode == 'leq':
					cycles = [cy for cy in nx.simple_cycles(G) if len(cy) <= path_length]
				elif mode == 'exact':
					cycles = [cy for cy in nx.simple_cycles(G) if len(cy) == path_length]
				
				used = []
				for cy in cycles:
					if len(cy) > 2:
						cy_set =  set(sorted(cy))
						if cy_set not in used:
							all_cycles.append(cy)
							used.append(cy_set)
			
			all_paths = all_paths + cycles

			for path in all_paths:

				fp_dict = fingerprints[len(path)]

				path_coords = [n[1]['coords'] for n in G.nodes(data=True) if n[0] in path]
				adsorbate_combinations = itertools.product(adsorbate, repeat=len(path))
				
				for ads_comb in adsorbate_combinations:

					adsonly_positions = []
					slab_coords = [(sl.symbol, sl.position) for sl in self.blank_slab_ase]

					for p, a in zip(path_coords, ads_comb):

						ads, shift_ind = a

						elems = []
						positions = []
						for atom in ads:
							elems.append(atom.symbol)
							positions.append(atom.position)
						elems = np.asarray(elems)
						positions = np.asarray(positions)
						trans = np.dot(unit_cell, np.asarray(s))
						positions -= positions[shift_ind]
						positions += p

						for e,c in zip(elems, positions):
							slab_coords.append((e, c))
							adsonly_positions.append(c)

					advance, index, sgs, sgn, dists, atoms, formula = redundancy_check(slab_coords, adsonly_positions, fp_dict, repeat_unit_cell, symmetry_tol)

					if advance:
						sg_counts[sgn] += 1
						fp_dict[sgn].append(dists)

					path_dict[formula + '_' + plane_string + '_' + str(len(path)) + '_' + str(ptype_counter) + '_' + sgs + '_' + index] = atoms

		self.path_configuration_dict = path_dict

def slab_directive_dynamics(mat_dict, by_layer=True, nlayers_frozen=1, by_elem=False, elems=['X'], tol=0.1):

	for key, atoms in mat_dict.iteritems():

		indices = []
		if by_layer:
			coords = list(atoms.positions)
			zs = sorted(list(set([np.round(c[2],3) for c in coords])))[0:nlayers_frozen]
			z_cutoff = max(zs) + tol
			indices.extend([atom.index for atom in atoms if atom.position[2] < z_cutoff])
		if by_elem:
			indices.extend([atom.index for atom in atoms if atom.symbol in elems])

		const = FixAtoms(indices=indices)
		atoms.set_constraint(const)

		mat_dict[key] = atoms

	return mat_dict

class bulk_adsorption_generator():

	def __init__(self, struct):

		ase_atoms = AseAtomsAdaptor.get_atoms(struct)

		self.bulk_pym = struct
		self.bulk_ase = ase_atoms
		self.minimal_unit_cell = ase_atoms.get_cell().T

	def make_supercell(self, replicate):

		r1,r2,r3 = replicate
		P = np.array([[r1,0,0],[0,r2,0],[0,0,r3]])
		self.bulk_ase = make_supercell(self.bulk_ase, P)
		self.duplications = replicate

	def Voronoi_tessalate(self):

		base_coords = [a.position for a in self.bulk_ase]
		repeat_unit_cell = self.bulk_ase.get_cell().T
		inv_ruc = np.linalg.inv(repeat_unit_cell)
		basis = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
		mesh = []

		for coord in base_coords:

			mesh.append(coord)
			
			fcoord = np.dot(inv_ruc, coord)
			zero_threshold_indices = fcoord < 1e-6
			fcoord[zero_threshold_indices] = 0.0
			one_threshold_indices = abs(fcoord - 1.0) < 1e-6
			fcoord[one_threshold_indices] = 0.0

			if np.all(fcoord):
				trans_vecs = [-1 * b for b in basis] + basis
			else:
				trans_vecs = [basis[dim] for dim in (0,1,2) if fcoord[dim] == 0.0]
				combs = list(itertools.combinations(trans_vecs, 2)) + list(itertools.combinations(trans_vecs, 3))
				for comb in combs:
					compound = np.array([0.0,0.0,0.0])
					for vec in comb:
						compound += vec
					trans_vecs.append(compound)

			for vec in trans_vecs:
				trans_coord = [np.round(i, 6) for i in np.dot(repeat_unit_cell, fcoord + vec)]
				mesh.append(trans_coord)

		mesh = np.asarray(mesh)
		vor = Voronoi(mesh)

		atoms = Atoms()
		for c in mesh:
			atoms.append(Atom('Cu', c))
		for v in vor.vertices:
			atoms.append(Atom('H', v))
		atoms.set_cell(repeat_unit_cell.T)
		write('check.xyz', atoms, format='xyz')

		return vor, mesh

	def enumerate_ads_config(self, adsorbate, loading, site='all', symmetry_tol=0.01):
		
		vor, mesh = self.Voronoi_tessalate()
		vcoords = vor.vertices
		base_coords = [a.position for a in self.bulk_ase]
		repeat_unit_cell = self.bulk_ase.get_cell().T
		inv_ruc = np.linalg.inv(repeat_unit_cell)

		corrected_vcoords = []
		used = []
		for coord in vcoords:
			if np.linalg.norm(coord) < 1e3:
				fcoord = np.dot(inv_ruc, coord)
				zero_threshold_indices = abs(fcoord) < 1e-6
				fcoord[zero_threshold_indices] = 0.0
				one_threshold_indices = abs(fcoord - 1.0) < 1e-6
				fcoord[one_threshold_indices] = 0.0
				
				advance = True
				for u in used:
					if np.allclose(fcoord, u):
						advance = False
						break
				
				if advance:
					corrected_vcoords.append(np.dot(repeat_unit_cell, fcoord))
					used.append(fcoord)

		vtypes = []
		for vert in corrected_vcoords:
			used = []
			spectrum = []
			for atom_coord in base_coords:
				fvert = np.dot(inv_ruc, vert)
				fatom_coord = np.dot(inv_ruc, atom_coord)
				fdist_vec, sym = PBC3DF_sym(fvert, fatom_coord)
				dist = np.linalg.norm(np.dot(repeat_unit_cell, fdist_vec))
				sym_atom_coord = np.dot(repeat_unit_cell, fatom_coord - sym)
				spectrum.append((dist,sym_atom_coord))

			spectrum.sort(key = lambda x: x[0])
			min_dist = spectrum[0][0]
			spectrum = [s[1] for s in spectrum if s[0] - min_dist < symmetry_tol] + [vert]

			spectrum_atoms = Atoms()
			for s in spectrum:
				spectrum_atoms.append(Atom('H', s))

			spectrum_atoms.set_cell(repeat_unit_cell.T)
			spectrum_struct = AseAtomsAdaptor.get_structure(spectrum_atoms)
			sga = SpacegroupAnalyzer(spectrum_struct, symprec=symmetry_tol)
			pgs = sga.get_point_group_symbol()
			symmetry_order = len(sga.get_point_group_operations())

			try:
				symmetry_type = symmetry_order_dict[symmetry_order]
			except KeyError:
				symmetry_type = 'unknown' + str(symmetry_order)

			vtypes.append((symmetry_type, vert))

		if site in ('all', 'single_site'):
			max_loading = float(len(vtypes))
		else:
			max_loading = float(len([ty for ty in vtypes if ty[0] == site]))

		all_combinations = [s for s in all_subsets(vtypes) if len(s) > 0]

		if isfloat(loading):

			fractional_loadings = [abs(len(s)/max_loading - loading) for s in all_combinations]
			closest = min(set(fractional_loadings))
			all_combinations = [all_combinations[i] for i in range(len(all_combinations)) if fractional_loadings[i] == closest]

		if site == 'single_site':
			site_dict = dict((k,[]) for k in set(ty[0] for ty in vtypes))
			for entry in vtypes:
				ty, coord = entry
				site_dict[ty].append(coord)
		elif site == 'all':
			site_dict = {'all':[]}
			for entry in vtypes:
				site_dict['all'].append(entry)
		else:
			site_dict = {site:[]}
			for entry in vtypes:
				ty, coord = entry
				if ty == site:
					site_dict[ty].append(coord)

		ads_dict = {}
		atype_counter = 0

		fingerprints = dict((len(s), dict((k,[]) for k in range(1,231))) for s in all_combinations)
		sg_counts    = dict((k,0) for k in range(1,231))		

		for subset in all_combinations:

			types  = [s[0] for s in subset]
			type_string = ''
			for ty in set(types):
				type_string += ty + '_'

			if len(set(types)) > 1 and site == 'single_site':
				continue
			elif site not in ('all', 'single_site'):
				if len(set(types)) > 1:
					continue
				else:
					if list(set(types))[0] != site:
						continue

			coords = [s[1] for s in subset]
			ld = len(subset)
			fp_dict = fingerprints[ld]

			fractional_loading = ld/max_loading
			adsorbate_combinations = itertools.product(adsorbate, repeat=ld)

			for ads_comb in adsorbate_combinations:

				adsonly_positions = []
				bulk_coords = [(sl.symbol, sl.position) for sl in self.bulk_ase]

				for s,a in zip(subset, ads_comb):

					ads, shift_ind = a
					ty, site_coord = s

					elems = []
					positions = []
					for atom in ads:
						elems.append(atom.symbol)
						positions.append(atom.position)
					elems = np.asarray(elems)
					positions = np.asarray(positions)
					positions -= positions[shift_ind]
					positions += site_coord

					for e,c in zip(elems, positions):
						bulk_coords.append((e, c))
						adsonly_positions.append(c)

				advance, index, sgs, sgn, dists, atoms, formula = redundancy_check(bulk_coords, adsonly_positions, fp_dict, repeat_unit_cell, symmetry_tol)

				if advance:
					sg_counts[sgn] += 1
					fp_dict[sgn].append(dists)

				ads_dict[formula + '_' + str(int(fractional_loading * 1000)) + '_' + type_string + str(sgn) + '_' + index] = atoms

		self.adsorbate_configuration_dict = ads_dict

	def write_empty_bulk(self, filetype='cif'):

		bulk = self.bulk_ase
		formula = bulk.get_chemical_formula()

		write(formula + '.' + filetype, bulk, format=filetype)

class specified_adsorption_generator():

	def __init__(self, struct):

		""" position should be 'mid_faces', 'mid_edges', 'corners', or an array of 
			coordinates where adsorbates should be placed
		""" 
		ase_atoms = AseAtomsAdaptor.get_atoms(struct)

		self.bulk_pym = struct
		self.bulk_ase = ase_atoms
		self.minimal_unit_cell = ase_atoms.get_cell().T

	def enumerate_ads_config(self, adsorbate, loading, positions='mid_faces', coord_type='fractional', symmetry_tol=0.01):

		unit_cell = self.minimal_unit_cell
		basis = [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]
		neg_basis  = [-1 * b for b in basis]

		if positions == 'mid_faces':
			
			ads_positions = [
			[0.5, 0.5, 0.0],
			[0.5, 0.0, 0.5],
			[0.0, 0.5, 0.5]
			]

			positions_string = positions
			max_loading = 3.0

		else:
			positions_string = 'custom'

		ld = len(ads_positions)
		all_combinations = [s for s in all_subsets(ads_positions) if len(s) > 0]

		if isfloat(loading):

			fractional_loadings = [abs(len(s)/max_loading - loading) for s in all_combinations]
			closest = min(set(fractional_loadings))
			all_combinations = [all_combinations[i] for i in range(len(all_combinations)) if fractional_loadings[i] == closest]

		ads_dict = {}
		fingerprints = dict((len(s), dict((k,[]) for k in range(1,231))) for s in all_combinations)
		sg_counts    = dict((k,0) for k in range(1,231))

		for subset in all_combinations:

			ld = len(subset)
			adsorbate_combinations = itertools.product(adsorbate, repeat=ld)
			fp_dict = fingerprints[ld]
			fractional_loading = ld/max_loading

			for ads_comb in adsorbate_combinations:
				
				adsonly_positions = []
				blank_coords = [(sl.symbol, sl.position) for sl in self.bulk_ase]
					
				for p, a in zip(ads_positions, ads_comb):

					if coord_type == 'fractional':
						p = np.dot(unit_cell, p)
					
					ads, shift_ind = a
	
					elems = []
					positions = []
					for atom in ads:
						elems.append(atom.symbol)
						positions.append(atom.position)
					elems = np.asarray(elems)
					positions = np.asarray(positions)
					trans = np.dot(unit_cell, np.asarray(p))
					positions -= positions[shift_ind]
					positions += p
					
					for e,c in zip(elems, positions):
						blank_coords.append((e, c))
						adsonly_positions.append(c)

				advance, index, sgs, sgn, dists, atoms, formula = redundancy_check(blank_coords, adsonly_positions, fp_dict, unit_cell, symmetry_tol)

				if advance:
					sg_counts[sgn] += 1
					fp_dict[sgn].append(dists)

				ads_dict[formula + '_' + str(int(fractional_loading * 1000)) + '_' + positions_string + str(sgn) + '_' + index] = atoms

		self.adsorbate_configuration_dict = ads_dict

### Testing
from materials_project_query import mp_query
from write_outputs import write_adsorption_configs

q = mp_query('ghLai1BTnNsvWZPu')
m = q.make_structures('V')[1]

H = Atoms('H', positions=[(0, 0, 0)])
N = Atoms('N', positions=[(0, 0, 0)])

a = bulk_adsorption_generator(m)
a.make_supercell((1,1,1))
a.Voronoi_tessalate()
a.enumerate_ads_config([(N,0)], 'all', 'all')
write_adsorption_configs(a.adsorbate_configuration_dict, filetype='vasp')
a.write_empty_bulk(filetype='vasp')	

a = specified_adsorption_generator(m)
a.enumerate_ads_config([(N,0)], 0.33333)
write_adsorption_configs(a.adsorbate_configuration_dict, filetype='vasp')


