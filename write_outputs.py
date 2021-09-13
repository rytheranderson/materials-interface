from ase.io import write

def write_adsorption_configs(config_dict, filetype='cif', suffix=''):

    for config in config_dict:
        write(config + suffix + '.' + filetype, config_dict[config], format=filetype)

def write_ase_atoms(atoms, filename, filetype='cif'):

    write(filename + '.' + filetype, atoms, format=filetype)