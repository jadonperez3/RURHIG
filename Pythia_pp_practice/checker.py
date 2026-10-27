import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import pyarrow as pa
    import awkward as ak
    import vector

    import matplotlib.pyplot as plt

    vector.register_awkward()
    return ak, np, pa, plt


@app.cell
def _(ak):
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

    return (to_jet_and_consitit_vectors,)


@app.cell
def _(ak, pa, to_jet_and_consitit_vectors):
    def read_data(file_name, **kwargs):
        buffer = pa.memory_map(file_name, "rb")
        table = pa.ipc.open_file(buffer).read_all()
        ak_array = ak.from_arrow(table, generate_bitmasks=True)
        jets, constituents = to_jet_and_consitit_vectors(ak_array)
        jets = ak.with_field(jets, ak.enforce_type(ak_array.jet_parton_id, "var * int32"), where="parton_id")
        return jets, constituents

    return (read_data,)


@app.cell
def _(ak):
    def to_flat_numpy(arr):
        """Flatten a jagged awkward array (any nesting depth) into a plain numpy array."""
        return ak.to_numpy(ak.flatten(arr, axis=None))

    return (to_flat_numpy,)


@app.cell
def _(read_data):
    py8 = "/home/jepro/RURHIG/datasets/Pythia_pp200GeV_Dijet_Analysis/Pythia8_pp200GeV_parton_matching.arrow"

    jets_, constituents_ = read_data(py8)
    return constituents_, jets_


@app.cell
def _(jets_):
    print(len(jets_.fields))
    return


@app.cell
def _(np, to_flat_numpy):
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

    return


@app.cell
def _(ak):
    def find_charge(jets, constituents, kappa):
        pt_mask = constituents.pt > 1.0
        masked_pt = ak.where(pt_mask, constituents.pt, 0.0)
        masked_charge = ak.where(pt_mask, constituents.charge, 0.0)
        jet_charge = ak.sum(masked_charge * masked_pt**kappa, axis = -1) / jets.pt**kappa

        return jet_charge

    return (find_charge,)


@app.cell
def _(jets_, np, plt, to_flat_numpy):
    #Mother Hadron vs. pt
    parton_id_flat = to_flat_numpy(jets_.parton_id)
    pt_flat = to_flat_numpy(jets_.pt)

    gluon_mask = parton_id_flat == 21
    up_mask = np.abs(parton_id_flat) == 2
    down_mask = np.abs(parton_id_flat) == 1
    unmatched_mask = parton_id_flat == 0

    plt.figure()
    plt.hist(pt_flat[gluon_mask], bins=50, range=(0, 20), histtype="step", label="gluon")
    plt.hist(pt_flat[up_mask], bins=50, range=(0, 20), histtype="step", label="up quark")
    plt.hist(pt_flat[down_mask], bins=50, range=(0, 20), histtype="step", label="down quark")
    plt.hist(pt_flat[unmatched_mask], bins=50, range=(0, 20), histtype="step", label="unmatched")
    plt.xlabel("Jet $p_T$ [GeV]")
    plt.legend()
    plt.title("Jet $p_T$ by mother parton type")

 
    plt.show()
    return down_mask, gluon_mask, unmatched_mask, up_mask


@app.cell
def _(
    constituents_,
    down_mask,
    find_charge,
    gluon_mask,
    jets_,
    plt,
    to_flat_numpy,
    unmatched_mask,
    up_mask,
):
    jc2 = find_charge(jets_, constituents_, 0.6)
    jet_charge_flat = to_flat_numpy(jc2)

    plt.figure()
    plt.hist(jet_charge_flat[gluon_mask], bins=50, range=(-2.0, 2.0), histtype="step", label="gluon")
    plt.hist(jet_charge_flat[up_mask], bins=50, range=(-2.0, 2.0), histtype="step", label="up quark")
    plt.hist(jet_charge_flat[down_mask], bins=50, range=(-2.0, 2.0), histtype="step", label="down quark")
    plt.hist(jet_charge_flat[unmatched_mask], bins=50, range=(-2.0, 2.0), histtype="step", label="unmatched")
    plt.xlabel(r"Jet Charge [$\kappa$ = 0.6]")
    plt.yscale('log')
    plt.legend()
    plt.title("Jet charge by mother parton type")
 
    plt.show()
    return


if __name__ == "__main__":
    app.run()
