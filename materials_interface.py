# this is currently just a series of examples for the different functionalities

from grid_images import material_grid
from read_inputs import mp_query
import enumerate_adsorption as EA
import enumerate_vacancies as EV
from ase import Atoms

api_key = open('.MP_api_key', 'r').read().replace('\n', '')

H = Atoms('H', positions=[(0, 0, 0)])
N = Atoms('N', positions=[(0, 0, 0)])
C = Atoms('C', positions=[(0, 0, 0)])

q = mp_query(api_key)
m = q.make_structures('Pd')[0]

a = EA.surface_adsorption_generator(m, plane=(1,0,0), slab_depth=2)
a.make_supercell((3,3,1))
a.enumerate_ads_config([(N,0)], 3)
mat = material_grid(a.adsorbate_configuration_dict)
grid = mat.build_grid(square_grids=True)
mat.write_grid('ex0.png')

a = EA.surface_adsorption_generator(m, plane=(1,0,0), slab_depth=2)
a.make_supercell((2,2,1))
a.enumerate_ads_config([(H,0),(N,0)], 2, name='bridge')
mat = material_grid(a.adsorbate_configuration_dict)
grid = mat.build_grid(square_grids=True)
mat.write_grid('ex1.png')

a = EA.surface_adsorption_generator(m, (1,1,1), 2, 2)
a.make_supercell((3,3,1))
a.enumerate_ads_chains([(C,0)], 1.615, 6)
mat = material_grid(a.path_configuration_dict)
grid = mat.build_grid(square_grids=True)
mat.write_grid('ex2.png')

a = EA.bulk_adsorption_generator(m)
a.Voronoi_tessalate()
a.enumerate_ads_config([(H,0)], 3)
mat = material_grid(a.adsorbate_configuration_dict)
grid = mat.build_grid(square_grids=True)
mat.write_grid('ex3.xyz')
