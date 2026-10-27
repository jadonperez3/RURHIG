import awkward as ak
import vector 
import pyarrow as pa

vector.register_awkward()


buffer = pa.memory_map("Pythia8_pp200GeV.arrow", "rb")
table = pa.ipc.open_file(buffer).read_all()
events = ak.from_arrow(table)

constituents = ak.zip(
    {
        "pt" : events.constit_pt,
        "eta" : events.constit_eta,
        "phi" : events.constit_phi,
        "E" : events.constit_e,
    },
    with_name="Momentum4D"
)

jets_p4 = ak.sum(constituents, axis=1)

jet_mass = jets_p4.mass

#print(jet_mass)

print("N jets:", len(jet_mass))
print("min/max mass:", ak.min(jet_mass), ak.max(jet_mass))