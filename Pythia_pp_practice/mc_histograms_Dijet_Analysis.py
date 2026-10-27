import numpy as np
import pyarrow as pa
import awkward as ak
import vector

import matplotlib.pyplot as plt

vector.register_awkward()


def to_jet_and_consitit_vectors(arr):
    jets = ak.zip(dict(
        pt = ak.enforce_type(arr.jet_pt, "var * float32"),
        eta = ak.enforce_type(arr.jet_eta, "var * float32"),
        phi = ak.enforce_type(arr.jet_phi, "var * float32"),
        e = ak.enforce_type(arr.jet_e, "var * float32"),
        weight = ak.enforce_type(arr.jet_weight, "var * float32"),
        ncharged = ak.enforce_type(arr.jet_ncharged, "var * uint8"),
        nconstituents = ak.enforce_type(arr.jet_nconstituents, "var * uint8"),
    ), with_name="Momentum4D")
    

    constituents = ak.zip({
    key : ak.enforce_type(
        arr[f"constit_{key}"], 
        "var * var * float32" if key != "charge" else "var * var * int8",
    ) 
    for key in ("pt", "eta", "phi", "e", "charge")
}, with_name="Momentum4D")
    
    jets_p4 = ak.sum(constituents, axis=2)

    jet_mass = jets_p4.mass

    jets = ak.with_field(jets, jet_mass, where="mass")

    return jets, constituents    

def read_data(file_name, **kwargs):
    buffer = pa.memory_map(file_name, "rb")
    table = pa.ipc.open_file(buffer).read_all()
    ak_array = ak.from_arrow(table, generate_bitmasks=True)
    jets, constituents = to_jet_and_consitit_vectors(ak_array)
    jets = ak.with_field(jets, ak.enforce_type(ak_array.jet_parton_id, "var * int32"), where="parton_id")
    return jets, constituents

def to_flat_numpy(arr):
    """Flatten a jagged awkward array (any nesting depth) into a plain numpy array."""
    return ak.to_numpy(ak.flatten(arr, axis=None))

def combine_antiparticles(jets):

    jets_flat = to_flat_numpy(jets.parton_id)
    ids, freq = np.unique(jets_flat, return_counts=True)

    unmatched = 0
    up = 0
    down = 0
    gluon = 0
    other = 0

    for integer, count in zip(ids, freq):
    
        if integer == 0:
            unmatched += count
        elif np.abs(integer) == 1:
            down += count
        elif np.abs(integer) == 2:
            up += count
        elif integer == 21:
            gluon += count
        else:
            other += count

    return [unmatched, up, down, gluon, other]

def find_charge(jets, constituents, kappa):
    pt_mask = constituents.pt > 1.0
    masked_pt = ak.where(pt_mask, constituents.pt, 0.0)
    masked_charge = ak.where(pt_mask, constituents.charge, 0.0)
    jet_charge = ak.sum(masked_charge * masked_pt**kappa, axis = -1) / jets.pt**kappa

    return jet_charge
    

if __name__ == "__main__":

    py8_file = "/home/jepro/RURHIG/datasets/Pythia_pp200GeV_Dijet_Analysis/parton_matching_con_0.2_jet_2.arrow"

    jets, constituents = read_data(py8_file)

#Basic Kinematic Data
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    axes[0, 0].hist(to_flat_numpy(jets.mass), bins=50, range=(0, 0.5), histtype="step")
    axes[0, 0].set_xlabel("Jet Mass [GeV]")

    axes[0, 1].hist(to_flat_numpy(jets.pt), bins=50, range=(0, 20), histtype="step")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("Jet $p_T$ [GeV] (Log Scale)")

    axes[0, 2].hist(to_flat_numpy(jets.eta), bins=50, range=(-1, 1), histtype="step")
    axes[0, 2].set_xlabel(r"Jet $\eta$")

    axes[0, 3].hist(to_flat_numpy(jets.phi), bins=50, range=(-np.pi, np.pi), histtype="step")
    axes[0, 3].set_xlabel(r"Jet $\phi$")

    axes[1, 0].hist(to_flat_numpy(constituents.mass), bins=50, range=(0, 15), histtype="step")
    axes[1, 0].set_xlabel(r"Constituent Mass")

    axes[1, 1].hist(to_flat_numpy(constituents.pt), bins=50, range=(0, 15), histtype="step")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("Constituent $p_t$ [GeV] (Log Scale)")

    axes[1, 2].hist(to_flat_numpy(constituents.eta), bins=50, range=(-1,1), histtype="step")
    axes[1, 2].set_xlabel(r"Constituent $\eta$")

    axes[1, 3].hist(to_flat_numpy(constituents.phi), bins=50, range=(-np.pi, np.pi), histtype="step")
    axes[1, 3].set_xlabel(r"Constituent $\phi$")

    fig.suptitle("PP Pythia Simulation Basic Data")
    fig.tight_layout()

    plt.close() ###

#Azimuthal Correlation
    padded_jets = ak.pad_none(jets, 1, axis=1, clip=True)
    leading_jet_phi = padded_jets.phi[:, 0]
    dphi = leading_jet_phi - jets.phi[:, 1:]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    dphi_flat = to_flat_numpy(ak.drop_none(dphi))
    dphi_folded = np.abs(dphi_flat)
    
    plt.figure()
    plt.hist(dphi_folded, bins=50, range=(0, np.pi), histtype="step")
    plt.xlabel(r"$\Delta\phi$ [rad]")
    plt.title("Dijet azimuthal correlation")

    plt.close() ###

#2D Histogram dphi vs pt
    leading_jet_pt = padded_jets.pt
    leading_pt_broadcast, dphi_broadcast = ak.broadcast_arrays(leading_jet_pt, dphi)
    pt_flat = to_flat_numpy(leading_pt_broadcast)

    plt.figure()
    plt.hist2d(
    pt_flat,       # jet pT flattened
    dphi_folded,   # corresponding delta phi
    bins=(50, 50),
    range=[(0, 60), (0, np.pi)],
    )
    plt.colorbar(label="Counts")
    plt.xlabel("Jet $p_T$ [GeV]")
    plt.ylabel(r"$|\Delta\phi(\gamma, jet)|$")
    plt.title(r"Dijet $\Delta\phi$ vs jet $p_T$")

    plt.close()  ###

#Girth
    deta = constituents.eta - jets.eta
    dphi = constituents.phi - jets.phi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap
    delta_R = np.sqrt(deta**2 + dphi**2)
    girth = ak.sum((constituents.pt / jets.pt) * delta_R, axis=-1)
    girth_flat = to_flat_numpy(girth)

    plt.figure()
    plt.hist(girth_flat, bins=50, range=(0, 0.4), histtype="step")
    plt.title("Girth")

    plt.show()

#ptD
    ptd = np.sqrt(ak.sum(constituents.pt**2, axis=-1))/ak.sum(constituents.pt, axis=-1)
    ptd_flat = to_flat_numpy(ptd)

    plt.figure()
    plt.hist(ptd_flat, bins=50, range=(0, 1), histtype="step")
    plt.title("$p_TD$")

    plt.show()

#Lesub
    padded_constit = ak.pad_none(constituents, 2, axis=2, clip=True)
    lesub = padded_constit.pt[:, :, 0] - padded_constit.pt[:, :, 1]
    lesub_flat = to_flat_numpy(lesub)

    plt.figure()
    plt.hist(lesub_flat, bins=50, range=(0, 15), histtype="step")
    plt.title("Lesub")

    plt.show()

#Jet Charge
    
    jc1 = find_charge(jets, constituents, 0.3)
    jc2 = find_charge(jets, constituents, 0.6)
    jc3 = find_charge(jets, constituents, 1.0)
    fig, axes = plt.subplots(1, 3, figsize = (12, 4))

    axes[0].hist(to_flat_numpy(jc1), bins = 50, range = (-1.5, 1.5), histtype="step")
    axes[0].set_xlabel(r"Jet Charge for $\kappa$ = 0.3")

    axes[1].hist(to_flat_numpy(jc2), bins = 50, range = (-1.5, 1.5), histtype="step")
    axes[1].set_xlabel(r"Jet Charge for $\kappa$ = 0.6")

    axes[2].hist(to_flat_numpy(jc3), bins = 50, range = (-1.5, 1.5), histtype="step")
    axes[2].set_xlabel(r"Jet Charge for $\kappa$ = 1.0")

    fig.suptitle(r"Jet Charge Distributions for $\kappa$ = 0.3, 0.6, 1.0")
    fig.tight_layout()

    plt.close() ###

#Frequency of Mother Hardons
    plt.figure()
    names = ["unmatched", "up", "down", "gluon", "other"]
    counts = combine_antiparticles(jets)
    plt.bar(names, counts)
    plt.title("Mother Parton Classifications")

    plt.show()

#Mother Hadron vs. pt
    parton_id_flat = to_flat_numpy(jets.parton_id)
    pt_flat = to_flat_numpy(jets.pt)

    gluon_mask = parton_id_flat == 21
    up_mask = np.abs(parton_id_flat) == 2
    down_mask = np.abs(parton_id_flat) == 1
    unmatched_mask = parton_id_flat == 0

    plt.figure()
    plt.hist(pt_flat[gluon_mask], bins=50, range=(15, 25), histtype="step", label="gluon")
    plt.hist(pt_flat[up_mask], bins=50, range=(15, 25), histtype="step", label="up quark")
    plt.hist(pt_flat[down_mask], bins=50, range=(15, 25), histtype="step", label="down quark")
    plt.hist(pt_flat[unmatched_mask], bins=50, range=(15, 25), histtype="step", label="unmatched")
    plt.xlabel("Jet $p_T$ [GeV]")
    plt.yscale("log")
    plt.legend()
    plt.title("Jet $p_T$ by mother parton type")

    plt.show()

#Mother Hadron vs. charge
    jet_charge_flat = to_flat_numpy(jc2)

    plt.figure()
    plt.hist(jet_charge_flat[gluon_mask], bins=50, range=(-2.0, 2.0), histtype="step", label="gluon")
    plt.hist(jet_charge_flat[up_mask], bins=50, range=(-2.0, 2.0), histtype="step", label="up quark")
    plt.hist(jet_charge_flat[down_mask], bins=50, range=(-2.0, 2.0), histtype="step", label="down quark")
    plt.hist(jet_charge_flat[unmatched_mask], bins=50, range=(-2.0, 2.0), histtype="step", label="unmatched")
    plt.xlabel(r"Jet Charge [$\kappa$ = 0.6]")
    plt.legend()
    plt.title("Jet charge by mother parton type")
 
    plt.show()


