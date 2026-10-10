import os
if os.getcwd().split("/")[-1] != "Project_03":
    os.chdir("source/Project_03")
# Following code is called for numpy array pretty printing
#from pyscf import gto
import numpy as np
from scipy.linalg import fractional_matrix_power
from pprint import pp  # Data Pretty Printer, as it says
from typing import Dict, Tuple  # only used in type infer
import forte2
from Common.rhf_molecule import RHFMolecule


#%% Import Molecule Class from previous project
class Molecule(RHFMolecule):
    def __init__(self):
        super().__init__()
        self.mo_coeff = NotImplemented  # type: np.ndarray
        self.mo_energy = NotImplemented  # type: np.ndarray
        self.eri_ao = NotImplemented  # type: np.ndarray
        self.eri_mo = NotImplemented  # type: np.ndarray
        self.energy_rhf = NotImplemented  # type: float
        self.energy_corr = NotImplemented  # type: float
        self.e_mp2 = NotImplemented  # type: float

        def get_eri_ao(self): 
            self.eri_ao = self.integral_eri()

        def get_rhf_energy(self):
            self.energy_rhf, self.dm = self.scf_process()

        def get_mo_coeff(self):
            self.mo_coeff = self.C
        
        def get_mo_energy(self):
            self.mo_energy = self.eps

        def get_eri_mo(self):
            self.get_eri_ao()
            self.get_mo_coeff()
            eri_ao = self.eri_ao.copy()
            C = self.mo_coeff.copy()
            eri_mo_int_1 = np.einsum("pqrs,pi->iqrs", eri_ao, C)
            eri_mo_int_2 = np.einsum("iqrs,qj->ijrs", eri_mo_int_1, C)
            eri_mo_int_3 = np.einsum("ijrs,rk->ijks", eri_mo_int_2, C)
            eri_mo = np.einsum("ijks,sl->ijkl", eri_mo_int_3, C)
            self.eri_mo = eri_mo
        
        def get_mp2_energy(self):
            self.scf_process()
            self.get_eri_mo()
            self.get_mo_energy()
            self.get_mo_coeff()
            eri_mo = self.eri_mo.copy()
            nocc = self.nocc
            eri_b = eri_mo[:nocc, :nocc, nocc:, nocc:]
            eri_b_as = eri_b.copy().swapaxes(1,3)
            eps = self.mo_energy.copy()
            denom = 1.0 / (eps[:nocc, None, None, None] + eps[None, :nocc, None, None] - eps[None, None, nocc:, None] - eps[None, None, None, nocc:])
            num = eri_b * (2 * eri_b - eri_b_as)
            self.energy_corr = np.einsum("ijab,ijab->", num, denom)
        
        def get_total_energy(self):
            if self.energy_rhf is not NotImplemented and self.energy_corr is not NotImplemented:
                self.e_mp2 = self.energy_rhf + self.energy_corr
            else:
                raise ValueError("RHF energy or correlation energy has not been computed.")

        def print_mp2_information(self):
            print("=== MP2 Calculation Information ===")
            self.get_mp2_energy()
            self.get_total_energy()
            print("RHF Energy:", self.energy_rhf)
            print("Correlation Energy (MP2):", self.energy_corr)
            print("Total MP2 Energy:", self.e_mp2)


if __name__ == "__main__":
    mol = Molecule()
    mol.construct_from_dat_file("/Users/samliao/Desktop/PyPro_projects/Py_Quan_Pro/Project_04/input/h2o/DZ")
    mol.obtain_mol_instance(basis="cc-pVDZ")
    mol.get_nocc()
    mol.print_mp2_information()
    
    
