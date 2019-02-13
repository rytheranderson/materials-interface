import warnings

def warn(*args, **kwargs):
    pass

warnings.warn = warn

from pymatgen import Structure, MPRester, Molecule
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

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

