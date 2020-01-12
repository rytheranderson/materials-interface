import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

from pymatgen import Structure, MPRester, Molecule
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write

import matplotlib.pyplot as plt
import numpy as np

metals = ['V', 'Pd', 'Ru', 'Co', 'Cu', 'Ni', 'Fe', 'Au', 'Ag']

class mp_query():

	def __init__(self, api_key):
		self.mpr = MPRester(api_key)

	def find_access_strings(self, search):

		data = self.mpr.get_data(search)
		material_ids = [datum['material_id'] for datum in data]
	
		return material_ids
	
	def mp_structure(self, material_id, standardize=True):

		struct = self.mpr.get_structure_by_material_id(material_id)

		if standardize:
			struct = SpacegroupAnalyzer(struct).get_conventional_standard_structure()
	
		return struct
	
	def make_structures(self, search):
		
		material_ids = self.find_access_strings(search)
	
		structures = []
		for ID in material_ids:
			struct = self.mp_structure(ID)
			structures.append(struct)
	
		return structures

	def search_summary(self, search, print_energies=True, barplot=False, write_files=False, format='cif'):
		
		data  = self.mpr.get_data(search)
		energies = []

		for datum in data:
			f  = datum['pretty_formula']
			sn = str(datum['spacegroup']['number'])
			ss = datum['spacegroup']['symbol']
			i  = datum['material_id']
			te = float(datum['energy'])
			ae = float(datum['energy_per_atom'])

			struct = self.mp_structure(i, standardize=False)
			ase_atoms = AseAtomsAdaptor.get_atoms(struct)
			composition = ase_atoms.get_chemical_formula()

			if write_files:
				write(f + '_' + sn + '.' + format, ase_atoms, format=format)

			energies.append((f, sn, ss, te, ae, composition))

		energies.sort(key = lambda x: x[4])
		
		if print_energies:
			print('formula spacegroup_number spacegroup_symbol total_energy energy_per_atom composition')
			for l in energies:
				print('{:7} {:<17} {:<17} {:<12.5f} {:<15f} {:<11}'.format(*l))

		if barplot:
			
			energies = np.asarray(energies)
			epa = np.array(map(float, energies[:,4]))
			epa -= min(epa)
			xticks = [i + '_' + j for i,j in energies[:,0:2]]
			ind = np.arange(len(energies))

			fig = plt.figure()
			p1 = plt.bar(ind, epa)
			plt.ylabel('Relative energy per atom / eV')
			plt.xlabel('Material')
			plt.xticks(ind, xticks, rotation=90)
			fig.set_size_inches(6.5,3)
			plt.savefig(search + '.tiff', dpi=300, bbox_inches='tight')

class file_read():

	""" class for reading common file formats """

	def __init__(self, filename):
		self.filename = filename

	def read_cif(self):

		""" reads cifs, returns both pymatgen structure and ase atoms object """

		cif = CifParser(self.filename)
		struct = cif.get_structures(primitive=False)[0]

		ase_atoms = AseAtomsAdaptor.get_atoms(struct)

		return struct, ase_atoms

	def read_POSCAR(self):

		""" reads POSCARS or CONTCARS, returns both pymatgen structure and ase atoms object """

		poscar = Poscar.from_file(self.filename, check_for_POTCAR=False)
		struct = poscar.structure

		ase_atoms = AseAtomsAdaptor.get_atoms(struct)

		return struct, ase_atoms

#testing 
#from pymatgen.analysis.adsorption import *
#import matplotlib.pyplot as plt
#from enumerate_surface_adsorption import adsorbate_surface
#from ase import Atoms
#from ase.visualize import view
#from grid_images import material_grid
#from write_outputs import write_adsorption_configs
#
#asf = AdsorbateSiteFinder(struct)
#ads_sites = asf.find_adsorption_sites()
#print ads_sites
#a = adsorbate_surface(struct, (1,0,0), bulk=False)
#a.make_supercell((1,1,1))
#N= Atoms('N', positions=[(0, 0, 0)])
#a.add_adsorbate_atoms( [(N, 0)], loading=1.0, name='hollow')
#print a.adsorbate_configuration_dict
#write_adsorption_configs(a.adsorbate_configuration_dict)
#mat = material_grid(a.adsorbate_configuration_dict)
#mat.build_grid()
#view(mat.grid)

#q = mp_query('ghLai1BTnNsvWZPu')
#structs = q.make_structures('N-V')
#q.search_summary('N-V', write_files=True, format='cif', barplot=True, write_dict=True)


		