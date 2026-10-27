import numpy as np
import pyarrow as pa
import awkward as ak
import vector

import matplotlib.pyplot as plt
import matplotlib.colors

vector.register_awkward()


def to_vectors(arr):
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

    photons = ak.zip({
        key: ak.enforce_type(
            arr[f"photon_{key}"],
            "var * float32",
        )   
        for key in ("pt", "eta", "phi", "e")
    }, with_name="Momentum4D")
    
    jets_p4 = ak.sum(constituents, axis=2)

    jet_mass = jets_p4.mass

    jets = ak.with_field(jets, jet_mass, where="mass")

    return jets, constituents, photons    

def read_data(file_name, **kwargs):
    buffer = pa.memory_map(file_name, "rb")
    table = pa.ipc.open_file(buffer).read_all()
    ak_array = ak.from_arrow(
        table, generate_bitmasks=True,
    )
    jets, constituents, photons = to_vectors(ak_array)
    return jets, constituents, photons

def to_flat_numpy(arr):
    """Flatten a jagged awkward array (any nesting depth) into a plain numpy array."""
    return ak.to_numpy(ak.flatten(arr, axis=None))
    

if __name__ == "__main__":

    py8_file =  "../datasets/Pythia_--200GeV_Photon_Coincidence/Pythia8_pp200GeV_photon_coincidence.arrow"

    jets, constituents, photons = read_data(py8_file)

#Photon Kinematics
    fig, axes = plt.subplots(1, 4, figsize=(15,4))

    axes[0].hist(to_flat_numpy(photons.pt), bins=50, range=(0, 50), histtype="step")
    axes[0].set_xlabel("Photon $p_T$ [GeV]")

    axes[1].hist(to_flat_numpy(photons.eta), bins=50, range=(-1, 1), histtype="step")
    axes[1].set_xlabel(r"Photon $\eta$")

    axes[2].hist(to_flat_numpy(photons.phi), bins=50, range=(-np.pi, np.pi), histtype="step")
    axes[2].set_xlabel(r"Photon $\phi$")

    axes[3].hist(to_flat_numpy(photons.e), bins=50, range=(0, 60), histtype="step")
    axes[3].set_xlabel("Photon Energy [GeV]")   

    fig.suptitle("Photon Kinematics Data")
    fig.tight_layout()

#Gamma-Jet Coincidence
    sorted_idx = ak.argsort(photons.pt, axis=1, ascending=False)
    photons_sorted = photons[sorted_idx]
    padded_photons = ak.pad_none(photons_sorted, 1, axis=1, clip=True)
    leading_photon_phi = padded_photons.phi[:, 0] # one value per event
    dphi = jets.phi - leading_photon_phi  # broadcasting: subtracts leading photon phi from each jet's phi
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    dphi_flat = to_flat_numpy(ak.drop_none(dphi))
    dphi_folded = np.abs(dphi_flat)

    plt.figure()
    plt.hist(dphi_folded, bins=50, range=(-5.0, 5.0), histtype="step")
    plt.xlabel(r"$\Delta\phi$ [rad]")
    plt.title("Gamma+Jet Coincidece")

#Basic Kinematics
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))

    axes[0, 0].hist(to_flat_numpy(jets.mass), bins=50, range=(0, 0.5), histtype="step")
    axes[0, 0].set_xlabel("Jet Mass [GeV]")

    axes[0, 1].hist(to_flat_numpy(jets.pt), bins=50, range=(5, 25), histtype="step")
    axes[0, 1].set_xlabel("Jet $p_T$ [GeV]")

    axes[0, 2].hist(to_flat_numpy(jets.eta), bins=50, range=(-1, 1), histtype="step")
    axes[0, 2].set_xlabel(r"Jet $\eta$")

    axes[0, 3].hist(to_flat_numpy(jets.phi), bins=50, range=(-np.pi, np.pi), histtype="step")
    axes[0, 3].set_xlabel(r"Jet $\phi$")

    axes[1, 0].hist(to_flat_numpy(constituents.mass), bins=50, range=(0, 15), histtype="step")
    axes[1, 0].set_xlabel(r"Constituent Mass")

    axes[1, 1].hist(to_flat_numpy(constituents.pt), bins=50, range=(0, 15), histtype="step")
    axes[1, 1].set_xlabel(r"Constituent $p_t$")

    axes[1, 2].hist(to_flat_numpy(constituents.eta), bins=50, range=(-1,1), histtype="step")
    axes[1, 2].set_xlabel(r"Constituent $\eta$")

    axes[1, 3].hist(to_flat_numpy(constituents.phi), bins=50, range=(-np.pi, np.pi), histtype="step")
    axes[1, 3].set_xlabel(r"Constituent $\phi$")

    fig.suptitle("Jet and Constituent Kinematics")
    fig.tight_layout() 

#2D Histogram dphi vs pt
    pt_flat = to_flat_numpy(jets.pt)
    plt.figure()
    plt.hist2d(
    pt_flat,       # jet pT flattened
    dphi_folded,   # corresponding delta phi
    bins=(50, 50),
    range=[(10, 60), (0, np.pi)],
    norm=matplotlib.colors.LogNorm(),
    )
    plt.colorbar(label="Counts")
    plt.xlabel(r"Jet $p_T$ [GeV]")
    plt.ylabel(r"$|\Delta\phi(\gamma, jet)|$")
    plt.title("Photon-jet $\Delta\phi$ vs jet $p_T$")

    plt.show()