import os
import json

import numpy as np
import pyarrow as pa
import awkward as ak
import vector

import matplotlib.pyplot as plt


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

    return jets, constituents    

def read_data(file_name, **kwargs):
    buffer = pa.memory_map(file_name, "rb")
    table = pa.ipc.open_file(buffer).read_all()
    ak_array = ak.from_arrow(
        table, generate_bitmasks=True,
    )
    jets, constituents = to_jet_and_consitit_vectors(ak_array)
    return jets, constituents
    
if __name__ == "__main__":
    py8_file =  "../datasets/Pythia_pp200GeV/Pythia8_pp200GeV.arrow"

    jets, constituents = read_data(py8_file)

    jets[0].show()
    constituents[0].show()

    plt.hist(ak.to_numpy(ak.flatten(jets.pt, axis=None)), histtype="step", bins=50, range=(0,50), label="Jets pT")
    
    plt.show()
    
 
