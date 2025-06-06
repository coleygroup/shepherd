# Co-factors from PDB structures

Best docking scores using AutoDock Vina 1.2.5 (exhaustiveness = 1000).

Generated the lowest energy conformers by:
1. Embed 1000 molecules using RDKit v2024.03.3 ETKDG
2. Cluster with RDKit's Butina clustering
3. Relax remaining structures with xTB v6.6.1 in implicit water as solvent
4. Cluster those structures with RDKit's Butina clustering
5. Select the conformer with the lowest energy

Order of structures:
0. Cc1ccc(OC[C@@H](O)[C@H](C)NC(C)C)c2c1CCC2
    - JRZ for 3ny8: -8.615 kcal/mol
1. CCCOc1cc(Cl)cc(-c2cc(-c3ccccc3C#N)cn(-c3cccnc3)c2=O)c1
    - XF1 for 7l11: -9.132
2. Nc1nc(NCCc2ccc(O)cc2)nc2nc(-c3ccco3)nn12
    - ZMA for 3eml: -9.158 kcal/mol
3. COc1cccc(-c2cc(-c3ccc(C(=O)O)cc3)c(C#N)c(=O)[nH]2)c1
    - QZZ for 4unn: -9.251 kcal/mol
4. O=C(C=Cc1ccc(O)cc1)c1ccc(O)cc1O
    - HCC for 4rlu: -8.697 kcal/mol
5. Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1
    - STI / Imatinib for 1iep: -9.521 kcal/mol
6. O=C(Nc1ccc(OC(F)(F)Cl)cc1)c1cnc(N2CC[C@@H](O)C2)c(-c2ccn[nH]2)c1
    - AY7 / asciminib for 5mo4: -9.938 kcal/mol

# Top 1 from docking screen of 10k random molecules selected from the test set of MOSES
`molblock_charges_docking_screen.pkl`

Followed the same procedure to find the lowest energy conformers as above.

Best docking scores using AutoDock Vina 1.2.5 (exhaustiveness = 32).

0. Cn1cc(NC(=O)c2ccc3c(c2)CC(c2ccccc2)OC3=O)cn1
    - 1iep: -12.11 kcal/mol
1. Cc1ccc2nc(-c3ccccn3)cc(C(=O)Nc3cccnc3)c2c1
    - 3eml: -11.341 kcal/mol
2. Cn1nc(C(=O)Nc2n[nH]c3ccccc23)c2ccccc2c1=O
    - 3ny8: -11.857 kcal/mol
3. Cc1cccc(C(=O)Nc2cccc(-c3nn4c(C)nnc4s3)c2)c1
    - 4rlu: -11.904 kcal/mol
4. Cc1cc(-c2cccc(-c3ccn(Cc4nnc(C)o4)n3)c2)cc(C)n1
    - 4unn: -10.636 kcal/mol
5. O=C(c1n[nH]c2ccccc12)N1CCc2c([nH]c3ccnn3c2=O)C1
    - 5mo4: -10.461 kcal/mol
6. O=C1CC(C(=O)N2CCc3cc(F)ccc3C2)c2ccccc2N1
    - 7l11: -8.815 kcal/mol


# PDB co-crystal ligands -- pose and lowest energies
- `molblock_charges_pdb_pose.pkl`
    - Added hydrogens with rdkit, not xtb optimized
- `molblock_charges_pdb_lowestenergy.pkl`
    - Lowest energy conformer of PDB co-crystal

Order (but check) [4unn, 7l11, 5mo4, 1iep, 3ny8, 4rlu, 3eml]

# Top 1 docked from docking screen -- pose and lowest energies
- `molblock_charges_bestdocked_pose.pkl`
    - Added hydrogens with rdkit, not xtb optimized
- `molblock_charges_bestdocked_lowestenergy.pkl
    - lowest energy conformers

In order of ['1iep', '3eml', '3ny8', '4rlu', '4unn', '5mo4', '7l11']
