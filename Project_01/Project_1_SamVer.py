import numpy as np
import os
from molmass import ELEMENTS, Element
from scipy import constants

##define physical constants( borrow from solution ）
E_h = constants.physical_constants["Hartree energy"][0]
a_0 = constants.physical_constants["Bohr radius"][0]
amu = constants.physical_constants["atomic mass constant"][0]
c_0 = constants.physical_constants["speed of light in vacuum"][0]
h = constants.physical_constants["Planck constant"][0]
kilo = constants.kilo
centi = constants.centi
angstrom = constants.angstrom
mega = constants.mega

ANGLE_EPS = 1.0e-10
BOND_LIMIT = 3.0  # Bohr

#%%
class Molecule:

    def __init__(self):
        self.atom_charges = NotImplemented  # type: np.ndarray
        self.atom_coords = NotImplemented  # type: np.ndarray
        self.natm = NotImplemented  # type: int
        self.com = NotImplemented  # type: np.ndarray

    def construct_from_dat_file(self, file_path: str):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        self.natm = int(lines[0].strip())
        self.atom_coords = np.array([list(map(float, line.split()[1:4])) for line in lines[1:self.natm + 1]], dtype=float)
        self.atom_charges = np.array([float(line.split()[0]) for line in lines[1:self.natm + 1]])
        
        def C_of_mass(self) -> np.ndarray:
            mr = np.zeros(3)
            sum_m = 0.0
            for i in range(self.natm):
                mr += ELEMENTS[self.atom_charges[i]].mass * self.atom_coords[i] 
                sum_m += ELEMENTS[self.atom_charges[i]].mass
            return mr / sum_m
        
        self.com = C_of_mass(self)
        # Input: Read file from `file_path`
        # Attribute modification: Obtain atom charges to `self.atom_charges`
        # Attribute modification: Obtain coordinates to `self.atom_coords`
        # Attribute modification: Obtain atom numbers in molecule to `self.natm`
        # raise NotImplementedError("Reader need to fill this method function!")
    
    def bond_length(self, atom1: int, atom2: int) -> float:
        return np.linalg.norm(self.atom_coords[atom1] - self.atom_coords[atom2])
    
    def unit_vector(self, atom1: int, atom2: int) -> np.ndarray:
        vec = self.atom_coords[atom2] - self.atom_coords[atom1]
        return vec / np.linalg.norm(vec)    
    
    def bond_angle(self, i: int, j: int, k: int) -> float:
        vec1 = self.unit_vector(j, i)
        vec2 = self.unit_vector(j, k)
        cos_angle = np.dot(vec1, vec2)
        return np.arccos(cos_angle) * 180.0 / np.pi 


    def is_valid_bond_angle(self,i,j,k) -> bool:
        return self.bond_length(i,j) < BOND_LIMIT and self.bond_length(j,k) < BOND_LIMIT and self.bond_angle(i,j,k) > 90

    def is_valid_out_of_plane_angle(
    self,
    i: int,
    j: int,
    k: int,
    l: int,
) -> bool:
    # i is the out-of-plane atom.
    # j-k-l defines the reference plane, with k as the center.
        if len({i, j, k, l}) != 4:
            return False

        if not all(0 <= atom < self.natm for atom in (i, j, k, l)):
            return False

        # All three atoms should be connected to the center atom k.
        if not (
            self.bond_length(k, i) < BOND_LIMIT
            and self.bond_length(k, j) < BOND_LIMIT
            and self.bond_length(k, l) < BOND_LIMIT
        ):
            return False

        e_kj = self.unit_vector(k, j)
        e_kl = self.unit_vector(k, l)

            # j-k-l must define a plane and therefore cannot be collinear.
        normal_norm = np.linalg.norm(np.cross(e_kj, e_kl))

        return normal_norm > 1.0e-10

        ##### Borrow ? don't know why
    def out_of_plane_angle(
        self,
        i: int,
        j: int,
        k: int,
        l: int,
    ) -> float:
        res = np.cross(self.unit_vector(k, j), self.unit_vector(k, l)).dot(self.unit_vector(k, i))
        res /= np.sin(self.bond_angle(j, k, l) / 180 * np.pi)
        np.clip(res,-1.0,1.0)
        assert(np.abs(res) < 1 + 1e-7)
        res = np.sign(res) if np.abs(res) > 1 else res
        return np.arcsin(res) * 180 / np.pi

    def is_valid_dihedral_angle(
    self, i: int, j: int, k: int, l: int
) -> bool:
        if len({i, j, k, l}) != 4:
            return False
        
        return (self.is_valid_bond_angle(i,j,k) or self.is_valid_bond_angle(i,k,j)) \
            and (self.is_valid_bond_angle(j,k,l) or self.is_valid_bond_angle(k,j,l)) \
            and self.bond_length(j,k) < BOND_LIMIT


    def dihedral_angle(
        self, i: int, j: int, k: int, l: int
    ) -> float:

        e_ji = self.unit_vector(j, i)
        e_jk = self.unit_vector(j, k)
        e_kj = self.unit_vector(k, j)
        e_kl = self.unit_vector(k, l)

        n1 = np.cross(e_ji, e_jk)
        n2 = np.cross(e_kj, e_kl)
        cos = np.clip(np.dot(n1, n2), -1.0, 1.0)
        cos /= np.sin(self.bond_angle(i, j, k) / 180 * np.pi) * np.sin(self.bond_angle(j, k, l) / 180 * np.pi)
        cos = np.sign(cos) if np.abs(cos) > 1 else cos
        return np.arccos(cos) * 180 / np.pi
    

    def print_bond_length(self):
        print("=== Bond Lengths ===")
        for i in range(self.natm):
            for j in range(i + 1, self.natm):
                print(f"{i:3d} - {j:3d}: {self.bond_length(i, j):10.5f} Bohr")
        
    def print_bond_angle(self):
        print("=== Angles ===")
        for i in range(self.natm):
            for j in range(i+1, self.natm):
                for k in range(j+1,self.natm):
                    if self.is_valid_bond_angle(i, j, k):
                        angle = self.bond_angle(i, j, k)
                        print(
                            f"{i:3d} - {j:3d} - {k:3d}: "
                            f"{angle:10.5f} Degree"
                        )

                    elif self.is_valid_bond_angle(j, i, k):
                        angle = self.bond_angle(j, i, k)
                        print(
                            f"{j:3d} - {i:3d} - {k:3d}: "
                            f"{angle:10.5f} Degree"
                        )

                    elif self.is_valid_bond_angle(i, k, j):
                        angle = self.bond_angle(i, k, j)
                        print(
                            f"{i:3d} - {k:3d} - {j:3d}: "
                            f"{angle:10.5f} Degree"
                        )

    def print_out_of_plane_angle(self):
        print("=== Out-of-Plane Angles ===")
        for j in range(self.natm):
            for k in range(j + 1, self.natm):
                for l in range(k + 1, self.natm):
                    for tup in [(j, k, l), (k, j, l), (j, l, k)]:
                        if self.is_valid_bond_angle(*tup):
                            for i in range(self.natm):
                                if i not in [j, k, l] and self.is_valid_out_of_plane_angle(i, *tup):
                                    print("{:3d} - {:3d} - {:3d} - {:3d}: {:10.5f} Degree".format(i, *tup, self.out_of_plane_angle(i, *tup)))


    def print_dihedral_angle(self):
        print("=== Dihedral Angles ===")

        for j in range(self.natm):
            for k in range(j + 1, self.natm):
                for i in range(self.natm):
                    for l in range(i + 1, self.natm):
                            if self.is_valid_dihedral_angle(i, j, k, l):
                                angle = self.dihedral_angle(i, j, k, l)
                                print(
                                    f"{i:3d} - {j:3d} - {k:3d} - {l:3d}: "
                                    f"{angle:10.5f} degree"
                                )

    def momentum_of_inertia(self) -> np.ndarray:
        atom_mass = np.array([ELEMENTS[self.atom_charges[i]].mass for i in range(self.natm)])
        coord_bar = self.atom_coords - self.com
        I = np.einsum("i,ix,iy -> xy", atom_mass, coord_bar, coord_bar)
        np.fill_diagonal(I, I.diagonal().sum() - I.diagonal())    ## borrow from the solution, very tricky way.
        return np.linalg.eigvalsh(I)
    
    def rotational_constant(self) -> np.ndarray:
        I = self.momentum_of_inertia()
        pi = np.pi
        return h/(8 * pi ** 2 * c_0 * I) / amu / a_0**2 * centi
    
    def print_all_info(self):
        print(f"Number of atoms: {self.natm}")
        print(f"Center of mass: {self.com}")
        self.print_bond_length()
        self.print_bond_angle()
        self.print_out_of_plane_angle()
        self.print_dihedral_angle()
        print(f"Momentum of inertia in bohr: {self.momentum_of_inertia()}")
        print(f"Rotational constant (in cm-1): {self.rotational_constant()}")

        print(f"Atom coordinates: {self.atom_coords}")

if __name__ == "__main__":
    mol = Molecule()
    mol.construct_from_dat_file(os.getcwd()+"/input/allene.dat")
    mol.print_all_info()