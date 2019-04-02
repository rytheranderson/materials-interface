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

symmetry_order_dict = {24:'tetra', 48:'fcc_octa', 8:'bcc'}

