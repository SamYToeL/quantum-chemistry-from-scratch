import numpy as np
import forte2


ELEMENTS = {
    1: "H",    2: "He",   3: "Li",   4: "Be",   5: "B",    6: "C",    7: "N",    8: "O",    9: "F",    10: "Ne",
    11: "Na",  12: "Mg",  13: "Al",  14: "Si",  15: "P",   16: "S",   17: "Cl",  18: "Ar",  19: "K",   20: "Ca",
    21: "Sc",  22: "Ti",  23: "V",   24: "Cr",  25: "Mn",  26: "Fe",  27: "Co",  28: "Ni",  29: "Cu",  30: "Zn",
    31: "Ga",  32: "Ge",  33: "As",  34: "Se",  35: "Br",  36: "Kr",  37: "Rb",  38: "Sr",  39: "Y",   40: "Zr",
    41: "Nb",  42: "Mo",  43: "Tc",  44: "Ru",  45: "Rh",  46: "Pd",  47: "Ag",  48: "Cd",  49: "In",  50: "Sn",
    51: "Sb",  52: "Te",  53: "I",   54: "Xe",  55: "Cs",  56: "Ba",  57: "La",  58: "Ce",  59: "Pr",  60: "Nd",
    61: "Pm",  62: "Sm",  63: "Eu",  64: "Gd",  65: "Tb",  66: "Dy",  67: "Ho",  68: "Er",  69: "Tm",  70: "Yb",
    71: "Lu",  72: "Hf",  73: "Ta",  74: "W",   75: "Re",  76: "Os",  77: "Ir",  78: "Pt",  79: "Au",  80: "Hg",
    81: "Tl",  82: "Pb",  83: "Bi",  84: "Po",  85: "At",  86: "Rn",  87: "Fr",  88: "Ra",  89: "Ac",  90: "Th",
    91: "Pa",  92: "U",   93: "Np",  94: "Pu",  95: "Am",  96: "Cm",  97: "Bk",  98: "Cf",  99: "Es",  100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
    111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og",
}



class RHFMolecule:
    def __init__(self):
        self.atom_charges = NotImplemented  # type: np.ndarray
        self.atom_coords = NotImplemented   # type: np.ndarray
        self.natm = NotImplemented          # type: int
        self.mol = NotImplemented           # type: forte2.system
        self.path_dict = NotImplemented     # type: Dict[str, str]
        self.nao = NotImplemented           # type: int
        self.charge = 0                     # type: int
        self.nocc = NotImplemented          # type: int
        self.eps = NotImplemented  # type: np.ndarray
        self.C = NotImplemented    # type: np.ndarray
        self.dm = NotImplemented   # type: np.ndarray
        self.rhf_energy = NotImplemented  # type: float

    def construct_from_dat_file(self, file_path: str):
        # Same to Project 01
        with open(file_path, "r") as f:
            dat = np.array([line.split() for line in f.readlines()][1:])
            self.atom_charges = np.array(dat[:, 0], dtype=float).astype(int)
            self.atom_coords = np.array(dat[:, 1:4], dtype=float)
            self.natm = self.atom_charges.shape[0]
    #%% Define additional information for the Molecule class
    # Using forte2
    def obtain_mol_instance(self,basis: str):
        mole = forte2.System(
            xyz="\n".join(
                f"{ELEMENTS[self.atom_charges[i]]}  {self.atom_coords[i, 0]} {self.atom_coords[i, 1]} {self.atom_coords[i, 2]}"
                for i in range(self.natm)
            ),
            basis_set=basis,
            unit="bohr"
        )
        mole.charge = self.charge
        mole.spin = 0
        self.mol = mole

    def obtain_nao(self):
        self.nao = self.mol.nbf
    
    def integral_ovlp(self):
        return forte2.integrals.overlap(self.mol)
    
    def integral_kinetic(self):
        return forte2.integrals.kinetic(self.mol)
    
    def integral_nuclear(self):
        return forte2.integrals.nuclear(self.mol)

    def get_core(self, mol):
        T = forte2.integrals.kinetic(mol)
        V_nuc = forte2.integrals.nuclear(mol)
        H_core_ao = T + V_nuc
        return H_core_ao
    
    def core_repulsion(self):
        return self.mol.nuclear_repulsion
    
    def integral_eri(self):
        return forte2.integrals.coulomb_4c(self.mol)
    
    def integral_ovlp_m1d2(self):
        from scipy.linalg import fractional_matrix_power
        S = self.integral_ovlp()
        return fractional_matrix_power(S, -0.5)
    
    def get_fock(self) -> np.array:
        h_core = self.get_core(self.mol)
        g_ao = self.integral_eri()
        eri_cont_1 = np.einsum("pqrs,rs->pq", g_ao, self.dm)
        eri_cont_2 = np.einsum("prqs,rs->pq", g_ao, self.dm)
        fock = h_core + eri_cont_1 - 0.5 * eri_cont_2
        return fock
    
    def solve_roothaan(self):
        from scipy.linalg import eigh
        eps, C = eigh(self.get_fock(), self.integral_ovlp())
        self.eps, self.C = eps, C


    def get_nocc(self):
        nocc = (self.atom_charges.sum() - self.charge) // 2
        self.nocc = nocc
        
    def make_dm(self) -> np.array:
        dm = 2 * self.C[:,:self.nocc] @ self.C[:,:self.nocc].T
        self.dm = dm

    def update_dm(self) -> np.array:
        if self.C is not NotImplemented:
            self.make_dm()
        else:
            raise ValueError("Coefficient matrix has not been initialized.")
        return self.dm
    
    def e_total(self):
        h_core = self.get_core(self.mol)
        fock = self.get_fock()
        e_ele = 0.5 * np.einsum("pq,pq->", self.dm, h_core + fock)
        e_nuc = self.core_repulsion()
        e_total = e_ele + e_nuc
        return e_total

    def scf_process(self, dm_guess: np.array = None, max_iter = 64, thresh_eng = 1e-10, thresh_dm = 1e-8):
        if dm_guess is None:
            self.get_nocc()
            E = 0.
            self.obtain_nao()
            dm = np.zeros((self.nao, self.nao),dtype=float)
            self.dm = dm
        else:
            self.dm = dm_guess
        e_next, dm_next = NotImplemented, NotImplemented
        e_prev, dm_prev = E, dm
        # max_iter, thresh_eng, thresh_dm = 64, 1e-10, 1e-8
        for iter in range(max_iter):
            # Solve the Roothaan equations to get the coefficients
            self.solve_roothaan()
            #use fock matrix to get the total electronic energy and update the density matrix
            e_next = self.e_total()
            # update density matrix using the new coefficients

            dm_next = self.update_dm()
            # Check convergence
            if iter > 0:
                eng_diff = abs(e_next - e_prev)
                dm_diff = np.linalg.norm(dm_next - dm_prev)
                print(f"Iteration {iter}: E = {e_next}, ΔE = {eng_diff}, ΔDM = {dm_diff}")
                if eng_diff < thresh_eng and dm_diff < thresh_dm:
                    self.rhf_energy, self.dm = e_next, dm_next
                    break
                e_prev, dm_prev = e_next, dm_next
                if iter == max_iter - 1:
                    self.rhf_energy, self.dm = e_next, dm_next
                    print("SCF did not converge within the maximum number of iterations.")
        return self.rhf_energy, self.dm
    
    def dipole_moment(self):
        from forte2 import integrals
        multipole1 = integrals.emultipole1(self.mol)
        mtp = np.stack(multipole1[1:])
        dm = self.dm
        mu = - np.einsum("pq,xpq->x", dm, mtp) + np.einsum("x,xi->i",self.atom_charges,self.atom_coords)
        return mu
    def print_information(self):
        e, dm = self.scf_process()
        print("=== SCF Process ===")
        print("Total Energy:", e)
        print("Density Matrix:\n", dm)
        print("Dipole moment (in A.U.):", self.dipole_moment())

            