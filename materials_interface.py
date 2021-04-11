# this is currently just a series of examples for the different functionalities

from grid_images import material_grid
from read_inputs import mp_query
from write_outputs import write_adsorption_configs
import enumerate_adsorption as EA
import enumerate_vacancies as EV
from ase import Atoms
from ase.io import read, write
from pymatgen.io.ase import AseAtomsAdaptor


H = Atoms('H', positions=[(0, 0, 0)])
N = Atoms('N', positions=[(0, 0, 0)])
C = Atoms('C', positions=[(0, 0, 0)])

q = mp_query('STRING')
m = q.make_structures('Pd')[1]
atoms = AseAtomsAdaptor.get_atoms(m)
write('check.cif', atoms, format='cif')

#a = EA.surface_adsorption_generator(m, plane=(1,0,0), slab_depth=2)
#a.make_supercell((2,2,1))
#a.enumerate_ads_config([(H,0)], 2, name='bridge')
#write_adsorption_configs(a.adsorbate_configuration_dict, filetype='cif', suffix='')

#a = EA.surface_adsorption_generator(m, plane=(1,0,0), slab_depth=2)
#a.make_supercell((2,2,1))
#a.enumerate_ads_config([(H,0),(N,0)], 2, name='bridge')
#write_adsorption_configs(a.adsorbate_configuration_dict, filetype='cif', suffix='')

#a = EA.surface_adsorption_generator(m, (1,1,1), 2, 2)
#a.make_supercell((3,3,1))
#a.enumerate_ads_chains([(C,0)], 1.615, 6)
#mat = material_grid(a.path_configuration_dict)
#grid = mat.build_grid(square_grids=True)
#mat.write_grid('Cchains.xyz')

#a = EA.bulk_adsorption_generator(m)
#a.Voronoi_tessalate()
#a.enumerate_ads_config([(H,0)], 3)
#print(a.adsorbate_configuration_dict)
#write_adsorption_configs(a.adsorbate_configuration_dict, filetype='cif')

#q = mp_query('STRING')
#m = q.make_structures('PdO')[1]
#vsg = EV.vacancy_structure_generator(m)
#vsg.make_supercell((1,1,1))
#vsg.make_vacancies(3, elements=['O'])
#mat = material_grid(vsg.vacancy_dict)
#write_adsorption_configs(vsg.vacancy_dict, filetype='cif')
