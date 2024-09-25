from openbabel import openbabel as ob
from openbabel import pybel as pb

import numpy as np


# def get_array_from_obmol(obmol):
#     # Iterate over the atoms in the molecule
#     atom_array = []
#     for atom in ob.OBMolAtomIter(obmol):
#         # Append the coordinates and the atomic number of the atom (on each row)
#         atom_array.append([atom.GetX(), atom.GetY(), atom.GetZ(), atom.GetAtomicNum()])    
   
#     # Centering the data
#     atom_array -= np.mean(atom_array, axis=0)
#     return atom_array

def get_array_from_pybelmol(pybelmol):
    # Iterate over the atoms in the molecule
    atom_array = []
    for atom in pybelmol.atoms:
        # Append the coordinates and the atomic number of the atom (on each row)
        atom_array.append([atom.coords[0], atom.coords[1], atom.coords[2], atom.atomicnum])    
   
    # Centering the data
    atom_array -= np.mean(atom_array, axis=0)
    return atom_array
    
# mol_file = "target.sdf"
# mol_file = "MolScore_examples/GraphGA/target.smi"

# obabel 
# obconversion = ob.OBConversion()
# obconversion.SetInFormat("sdf")
# obmol = ob.OBMol()
# obconversion.ReadFile(obmol, "target.sdf")
# print(type(obmol))


# pybel 
obmol = next(pb.readfile("sdf", "target.sdf"))
print(type(obmol))



array = get_array_from_pybelmol(obmol)
print(array) 
# array = get_array_from_obmol(obmol)

# array = get_array_from_obmol(obmol)
 

# # mol = next(pb.readfile("smi", mol_file))
# smiles = ['CCO', 'CCN', 'CCOC', 'CCCCCC']
# for smile in smiles:
#     mol = pb.readstring("smi", smile)
#     mol.addh()
#     mol.make3D()
#     mol.localopt()
#     print(mol)
    



  

