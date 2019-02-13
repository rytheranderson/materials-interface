from ase.io import write

def write_adsorption_configs(config_dict, filetype='cif', prefix='slab'):

	for config in config_dict:
		write(config + '.' + filetype, ads_dict[config], format=filetype)
		