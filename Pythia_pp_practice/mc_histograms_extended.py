
import numpy as np
import pyarrow as pa
import awkward as ak
import vector
import matplotlib.pyplot as plt

vector.register_awkward()


def to_jet_and_constit_vectors(arr):
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
        key: ak.enforce_type(
            arr[f"constit_{key}"],
            "var * var * float32" if key != "charge" else "var * var * int8",
        )
        for key in ("pt", "eta", "phi", "e", "charge")
    }, with_name="Momentum4D")

    return jets, constituents


def read_data(file_name):
    buffer = pa.memory_map(file_name, "rb")
    table = pa.ipc.open_file(buffer).read_all()
    ak_array = ak.from_arrow(table, generate_bitmasks=True)
    return to_jet_and_constit_vectors(ak_array)


def to_flat_numpy(arr):
    """Flatten a jagged awkward array (any nesting depth) into a plain numpy array."""
    return ak.to_numpy(ak.flatten(arr, axis=None))


if __name__ == "__main__":
    py8_file = "Pythia8_pp200GeV.arrow"
    jets, constituents = read_data(py8_file)


    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].hist(to_flat_numpy(jets.pt), bins=50, range=(0, 20), histtype="step")
    axes[0, 0].set_xlabel("Jet $p_T$ [GeV]")

    axes[0, 1].hist(to_flat_numpy(jets.eta), bins=50, range=(-1, 1), histtype="step")
    axes[0, 1].set_xlabel(r"Jet $\eta$")

    axes[0, 2].hist(to_flat_numpy(jets.phi), bins=50, range=(-np.pi, np.pi), histtype="step")
    axes[0, 2].set_xlabel(r"Jet $\phi$")

    axes[1, 0].hist(to_flat_numpy(jets.mass), bins=50, range=(0, 10), histtype="step")
    axes[1, 0].set_xlabel("Jet mass [GeV]")

    axes[1, 1].hist(to_flat_numpy(jets.ncharged), bins=range(0, 15), histtype="step")
    axes[1, 1].set_xlabel("Charged constituents per jet")

    axes[1, 2].hist(to_flat_numpy(jets.nconstituents), bins=range(0, 15), histtype="step")
    axes[1, 2].set_xlabel("All constituents per jet")

    fig.suptitle("Per-jet kinematics (no cuts)")
    fig.tight_layout()


    n_jets_per_event = ak.num(jets, axis=1)
    plt.figure()
    plt.hist(ak.to_numpy(n_jets_per_event), bins=range(0, 8), histtype="step")
    plt.xlabel("Jets per event")
    plt.title("Jet multiplicity")


    pt_min = 3.0
    phi_lo, phi_hi = -np.pi / 2, np.pi / 2

    pt_cut_mask = jets.pt > pt_min
    phi_cut_mask = (jets.phi > phi_lo) & (jets.phi < phi_hi)

    pt_cut_jets = jets[pt_cut_mask]
    phi_cut_jets = jets[phi_cut_mask]
    both_cuts_jets = jets[pt_cut_mask & phi_cut_mask]

    plt.figure()
    plt.hist(to_flat_numpy(jets.pt), bins=50, range=(0, 20), histtype="step", label="no cuts")
    plt.hist(to_flat_numpy(both_cuts_jets.pt), bins=50, range=(0, 20), histtype="step",
              label=f"pt>{pt_min}, {phi_lo:.2f}<phi<{phi_hi:.2f}")
    plt.xlabel("Jet $p_T$ [GeV]")
    plt.legend()
    plt.title("Effect of pT + phi cuts (combined)")


    padded = ak.pad_none(jets, 2, axis=1, clip=True)
    leading_pt = padded.pt[:, 0]
    subleading_pt = padded.pt[:, 1]

    leading_pt_flat = ak.to_numpy(ak.drop_none(leading_pt))
    subleading_pt_flat = ak.to_numpy(ak.drop_none(subleading_pt))

    plt.figure()
    plt.hist(leading_pt_flat, bins=50, range=(0, 20), histtype="step", label="Leading jet")
    plt.hist(subleading_pt_flat, bins=50, range=(0, 20), histtype="step", label="Subleading jet")
    plt.xlabel("Jet $p_T$ [GeV]")
    plt.legend()
    plt.title("Leading vs subleading jet pT")


    dphi = padded.phi[:, 0] - padded.phi[:, 1]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    dphi_flat = ak.to_numpy(ak.drop_none(dphi))

    plt.figure()
    plt.hist(dphi_flat, bins=50, range=(-np.pi, np.pi), histtype="step")
    plt.xlabel(r"$\Delta\phi$(leading, subleading) [rad]")
    plt.title("Dijet azimuthal correlation")


    a_j = (leading_pt - subleading_pt) / (leading_pt + subleading_pt)
    a_j_flat = ak.to_numpy(ak.drop_none(a_j))

    plt.figure()
    plt.hist(a_j_flat, bins=50, range=(0, 1), histtype="step")
    plt.xlabel(r"$A_J = (p_{T,1} - p_{T,2}) / (p_{T,1} + p_{T,2})$")
    plt.title("Dijet pT asymmetry")


    frag_z = constituents.pt / jets.pt
    frag_z_flat = to_flat_numpy(frag_z)

    plt.figure()
    plt.hist(frag_z_flat, bins=50, range=(0, 1), histtype="step")
    plt.xlabel(r"$z = p_{T,\mathrm{constituent}} / p_{T,\mathrm{jet}}$")
    plt.title("Fragmentation function")


    charged_constit_pt = ak.where(constituents.charge != 0, constituents.pt, 0.0)
    charged_pt_sum = ak.sum(charged_constit_pt, axis=-1)   # shape (event, jet)
    charged_pt_frac = charged_pt_sum / jets.pt              # shape (event, jet)
    charged_pt_frac_flat = to_flat_numpy(charged_pt_frac)

    plt.figure()
    plt.hist(charged_pt_frac_flat, bins=50, range=(0, 1), histtype="step")
    plt.xlabel("Charged $p_T$ fraction per jet")
    plt.title("Charged pT fraction")


    padded = ak.pad_none(jets, 2, axis=1, clip=True)
    cut_padded = ak.pad_none(both_cuts_jets, 2, axis=1, clip=True)
    cut_leading_pt = ak.to_numpy(ak.drop_none(cut_padded.pt[:, 0]))
    cut_subleading_pt = ak.to_numpy(ak.drop_none(cut_padded.pt[:, 1]))

    plt.figure()
    plt.hist(to_flat_numpy(pt_cut_jets.pt), bins=50, range=(0, 20), histtype="step",
              label=f"pt > {pt_min}")
    plt.hist(to_flat_numpy(both_cuts_jets.pt), bins=50, range=(0, 20), histtype="step",
              label=f"pt>{pt_min}, {phi_lo:.2f}<phi<{phi_hi:.2f}")
    plt.hist(cut_leading_pt, bins=50, range=(0, 20), histtype="step",
              label="Leading jet (pt+phi cut)")
    plt.hist(cut_subleading_pt, bins=50, range=(0, 20), histtype="step",
              label="Subleading jet (pt+phi cut)")
    plt.xlabel("Jet $p_T$ [GeV]")
    plt.legend()
    plt.title("Jet pT: cuts and leading/subleading overlay")

    plt.show()
