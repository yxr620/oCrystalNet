from argparse import Namespace
from typing import List, Tuple
import numpy as np

import torch
from chemprop.data import CrystalDatapoint, CrystalDataset


# Memoization
CRYSTAL_TO_GRAPH = {}
ATOM_FDIM = 1494
# ofm 1056
# one hot 92
# element 438
# sapce feature 8
BOND_FDIM = 51
# this is the ini vector dimention of atom and bond
# 

def clear_cache():
    """Clears featurization cache."""
    global SMILES_TO_GRAPH
    SMILES_TO_GRAPH = {}


def get_atom_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of atom features.

    :param: Arguments.
    """
    return args.initdim
    # return ATOM_FDIM


def get_bond_fdim(args: Namespace) -> int:
    """
    Gets the dimensionality of bond features.

    :param: Arguments.
    """
    return BOND_FDIM


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self,
                 crystal_point: CrystalDatapoint,
                 args: Namespace):
        """
        Computes the graph structure and featurization of a molecule.

        :param crystal_point: a CrystalDatapoint object
        :param args: Arguments.
        """
        self.name = crystal_point.name
        self.crystal = crystal_point.crystal # This is a special structure of crystal
        self.n_atoms = len(self.crystal)   # number of atoms
        self.n_bonds = 0                   # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features, the feature in atom_init.json
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []      # mapping from atom index to incoming bond indices
        self.b2a = []      # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []   # mapping from bond index to the index of the reverse bond
        # print("crystal:")
        # print(self.crystal)

        # Get atom features
        for _ in range(self.n_atoms):
            self.a2b.append([])
        self.f_atoms = crystal_point.atom_features
        # Get bond features
        # print(crystal_point.point_indices)
        # print(np.shape(crystal_point.bond_features))
        # exit()
        for a1 in range(self.n_atoms):
            point_idxs = crystal_point.point_indices[a1, :] #len: 8
            bond_features = crystal_point.bond_features[a1, :, :] #len 8 * 54

            for a2, bond_feature in zip(point_idxs, bond_features):
                if args.atom_messages: # atom_message is false
                    self.f_bonds.append(self.f_atoms[a1].tolist() + bond_feature.tolist())
                    self.f_bonds.append(self.f_atoms[a2].tolist() + bond_feature.tolist())
                else:
                    self.f_bonds.append(bond_feature.tolist()) # this is the distance between two node (expanding it to vector)
                    self.f_bonds.append(bond_feature.tolist())

                # Update index mappings
                b1 = self.n_bonds # why b2 is the reverse of b1
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: List[MolGraph], args: Namespace):
        # get feature dim
        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = get_bond_fdim(args) + args.atom_messages * self.atom_fdim  # * 2

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1   # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1   # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]                        # mapping from atom index to incoming bond indices
        b2a = [0]                         # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]                      # mapping from bond index to the index of the reverse bond


        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])  # if b != -1 else 0

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

        # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the space feature, atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a


def mol2graph(crystal_batch: CrystalDataset, args: Namespace) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param crystal_batch: a list of CrystalDataset
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    crystal_graphs = list()
    for crystal_point in crystal_batch:
        if crystal_point in CRYSTAL_TO_GRAPH.keys():
            crystal_graph = CRYSTAL_TO_GRAPH[crystal_point]
        else:
            crystal_graph = MolGraph(crystal_point, args)
            if not args.no_cache and len(CRYSTAL_TO_GRAPH) <= 10000:
                CRYSTAL_TO_GRAPH[crystal_point] = crystal_graph
        crystal_graphs.append(crystal_graph)

    return BatchMolGraph(crystal_graphs, args)
