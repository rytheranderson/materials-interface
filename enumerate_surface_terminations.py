import warnings

def warn(*args, **kwargs):
	pass

warnings.warn = warn

from pymatgen import Structure
from pymatgen.core.surface import generate_all_slabs
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atom, Atoms
from ase.build import make_supercell
from ase.constraints import FixAtoms
from ase.io import write

class surface_termination_generator():

	def __init__(self, struct):

		ase_atoms = AseAtomsAdaptor.get_atoms(struct)

		self.struct = struct
		self.ase_atoms = ase_atoms
		self.minimal_unit_cell = ase_atoms.get_cell().T

	def enumerate_surfaces(self, max_index, slab_depth, vacuum_size, plane='all', replicate=(1,1,1), max_normal_search=5, symmetrize=True, tol=0.1):

		surface_dict = {}
		max_index = int(max_index)
		slabs = generate_all_slabs(self.struct, 
						   min_slab_size=slab_depth, 
						   min_vacuum_size=vacuum_size, 
						   max_index=max_index, 
						   in_unit_planes=True, 
						   center_slab=True,
						   symmetrize=symmetrize,
						   max_normal_search=max_normal_search,
						   tol=tol)

		print 'Surfaces enumerated.'

		if plane != 'all':
			slabs = [slab for slab in slabs if slab.miller_index==plane]

		count = 0
		for slab in slabs:
			count += 1
			r1,r2,r3 = replicate
			P = np.array([[r1,0,0],[0,r2,0],[0,0,r3]])
			ase_slab = AseAtomsAdaptor.get_atoms(slab)
			ase_slab = make_supercell(ase_slab, P)
			formula = ase_slab.get_chemical_formula()
			plane = slab.miller_index
			plane_string = ''.join(map(str, plane))
			surface_dict[formula + '_' + plane_string + '_' + str(count)] = ase_slab

		self.surface_dict = surface_dict

from read_inputs import mp_query, file_read
from write_outputs import write_adsorption_configs

m = file_read('MOR_SI.cif')
m, a = m.read_cif()
stg = surface_termination_generator(m)
stg.enumerate_surfaces(1, 1, 1, replicate=(1,1,1))
write_adsorption_configs(stg.surface_dict, filetype='vasp')


