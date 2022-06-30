from curses import KEY_SELECT
from jarvis.db.figshare import data
from jarvis.io.vasp.inputs import Poscar
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from tqdm import tqdm


import pandas as pd
import math


dft_3d = data(dataset='dft_3d')
print (len(dft_3d))
counter = 0

# get all the poscar file
# for i in dft_3d:
    # atoms = Atoms.from_dict(i['atoms'])
    # poscar = Poscar(atoms)
    # jid = i['jid']
    # filename = 'POSCAR-'+jid+'.vasp'
    # poscar.write_file(filename)

# get all the bandgap and formation energy
# l = []
# key_set = set()
# for i in tqdm(dft_3d):

#     counter += 1
#     print(i["jid"]) # crystal name
#     # print(i.keys())
#     print(i["optb88vdw_bandgap"]) # this is bandgap
#     print(i["formation_energy_peratom"])
#     print(f"mbj: {i['mbj_bandgap']}")
#     if counter == 10:
#         exit()
#     jid = i['jid']
#     if jid not in key_set:
#         # print(i["optb88vdw_bandgap"])
#         l.append(['POSCAR-' + jid, i["optb88vdw_bandgap"], i["formation_energy_peratom"]
#         ])

#     key_set.add(jid)

# table = pd.DataFrame(l, columns=["name", "band_gap", "formation_energy"])
# table.to_csv("jarvis.csv", index=None)



# get all the modulus
# l = []
# key_set = set()
# for i in tqdm(dft_3d):
#     # counter += 1
#     # print(i.keys())
#     # print(i["jid"]) # crystal name
#     # print(type(i["bulk_modulus_kv"]))
#     # if counter == 10:
#     #     exit()

#     jid = i['jid']
#     if jid not in key_set and i["bulk_modulus_kv"] != "na" and i["shear_modulus_gv"] != "na" and i["shear_modulus_gv"] > 0 and i["bulk_modulus_kv"] > 0:
#         # print(i["bulk_modulus_kv"])
#         # print(i["shear_modulus_gv"])
#         l.append(['POSCAR-' + jid, math.log10(i["bulk_modulus_kv"]), math.log10(i["shear_modulus_gv"])])
#     key_set.add(jid)
# table = pd.DataFrame(l, columns=["name", "K", "G"])
# table.to_csv("elastic.csv", index=None)



# Download all the band gap here
# l = []
# key_set = set()
# for i in tqdm(dft_3d):

#     # counter += 1
#     # print(i["jid"]) # crystal name
#     # # print(i.keys())
#     # print(i["optb88vdw_bandgap"]) # this is bandgap
#     # print(i["formation_energy_peratom"])
#     # if counter == 10:
#     #     exit()
#     jid = i['jid']
#     if jid not in key_set:
#         # print(i["optb88vdw_bandgap"])
#         # print(i["bulk_modulus_kv"])
#         l.append(['POSCAR-' + jid, i["optb88vdw_bandgap"], i["formation_energy_peratom"]
#         ])
#     key_set.add(jid)

# table = pd.DataFrame(l, columns=["name", "band_gap", "formation_energy"])
# table.to_csv("jarvis_full.csv", index=None)


# get all modulus without log
# l = []
# key_set = set()
# for i in tqdm(dft_3d):
#     jid = i['jid']
#     if jid not in key_set and i["bulk_modulus_kv"] != "na" and i["shear_modulus_gv"] != "na" and i["shear_modulus_gv"] > 0 and i["bulk_modulus_kv"] > 0:
#         # print(i["bulk_modulus_kv"])
#         # print(i["shear_modulus_gv"])
#         l.append(['POSCAR-' + jid, i["bulk_modulus_kv"], i["shear_modulus_gv"]])
#     key_set.add(jid)
# table = pd.DataFrame(l, columns=["name", "K", "G"])
# table.to_csv("elastic.csv", index=None)

l = []
key_set = set()
for i in tqdm(dft_3d):

    jid = i['jid']
    if jid not in key_set and i['mbj_bandgap'] != "na":
        # print(i["optb88vdw_bandgap"])
        l.append(['POSCAR-' + jid, i["mbj_bandgap"]])

    key_set.add(jid)

table = pd.DataFrame(l, columns=["name", "band_gap"])
table.to_csv("mbj.csv", index=None)


