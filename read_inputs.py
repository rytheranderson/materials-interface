import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

from pymatgen import Structure, MPRester, Molecule
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar
from pymatgen.io.cif import CifFile
from pymatgen.io.ase import AseAtomsAdaptor

class mp_query():

	def __init__(self, api_key):
		self.mpr = MPRester(api_key)

	def find_access_strings(self, search):

		data = self.mpr.get_data(search)
		material_ids = [datum['material_id'] for datum in data]
	
		return material_ids
	
	def mp_structure(self, material_id):

		struct = self.mpr.get_structure_by_material_id(material_id)
		struct = SpacegroupAnalyzer(struct).get_conventional_standard_structure()
	
		return struct
	
	def make_structures(self, search):
		material_ids = self.find_access_strings(search)
	
		structures = []
		for ID in material_ids:
			struct = self.mp_structure(ID)
			structures.append(struct)
	
		return structures

class file_read():

	""" class for reading common file formats """

	def __init__(self, filename):
		self.filename = filename

	def read_cif(self, standard_cell=True):

		""" reads cifs, returns both pymatgen structure and ase atoms object """

		cif = CifFile.from_file(self.filename)
		struct = cif.structure

		if standard_cell:
			struct = struct.get_conventional_standard_structure()

		ase_atoms = AseAtomsAdaptor.get_atoms(struct)

		return struct, ase_atoms

	def read_POSCAR(self, standard_cell=True):

		""" reads POSCARS or CONTCARS, returns both pymatgen structure and ase atoms object """

		poscar = Poscar.from_file(self.filename, check_for_POTCAR=False)
		struct = poscar.structure

		if standard_cell:
			struct = struct.get_conventional_standard_structure()

		ase_atoms = AseAtomsAdaptor.get_atoms(struct)

		return struct, ase_atoms

foo = file_read('foo.cif')
struct = foo.read_cif()

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
		