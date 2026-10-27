from functools import cache
from copy import deepcopy

import numpy as np
import numba as nb

import pyarrow as pa
import awkward as ak
import vector
import fastjet as fj

import pythia8mc as pythia
from pdgid import charge

vector.register_awkward()

# Event-level (one value per event row)
event_col_fields = {
    "event_weight": pa.field("event_weight", pa.float32()),
    "event_id": pa.field("event_id", pa.int64()),
}

# Jet-level (lists per event)
jet_col_fields = {
    "weight": pa.field("jet_weight", pa.list_(pa.float32())),
    "pt": pa.field("jet_pt", pa.list_(pa.float32())),
    "eta": pa.field("jet_eta", pa.list_(pa.float32())),
    "phi": pa.field("jet_phi", pa.list_(pa.float32())),
    "e": pa.field("jet_e", pa.list_(pa.float32())),
    "ncharged": pa.field("jet_ncharged", pa.list_(pa.uint8())),
    "nconstituents": pa.field("jet_nconstituents", pa.list_(pa.uint8())),
    "parton_id": pa.field("jet_parton_id", pa.list_(pa.int32())),
}

# Constituent-level (list-of-lists per event: jets -> constituents)
constit_col_fields = {
    "pt": pa.field("constit_pt", pa.list_(pa.list_(pa.float32()))),
    "eta": pa.field("constit_eta", pa.list_(pa.list_(pa.float32()))),
    "phi": pa.field("constit_phi", pa.list_(pa.list_(pa.float32()))),
    "e": pa.field("constit_e", pa.list_(pa.list_(pa.float32()))),
    "charge": pa.field("constit_charge", pa.list_(pa.list_(pa.int8()))),
}

hard_parton_col_fields = {
    "pt": pa.field("hard_parton_pt", pa.list_(pa.float32())),
    "eta": pa.field("hard_parton_eta", pa.list_(pa.float32())),
    "phi": pa.field("hard_parton_phi", pa.list_(pa.float32())),
    "id": pa.field("hard_parton_id", pa.list_(pa.int32())),

}

@cache
def get_schema(metadata=None):
    event_fields = list(event_col_fields.values())
    jet_fields = list(jet_col_fields.values())
    constit_fields = list(constit_col_fields.values())
    parton_fields = list(hard_parton_col_fields.values())
    schema = pa.schema(event_fields + jet_fields + constit_fields + parton_fields, metadata=metadata)
    print("Pyarrow schema for writing:", schema)
    return schema


def events_to_rb_dict(
    events,
    jets: ak.Array,
    constituents: ak.Array,
    hard_partons: ak.Array,
    event_offset: int = 0,
) -> pa.RecordBatch:
    rec = {}
    n_events = len(jets)  # jets is shape (n_events, n_jets)

    # --- event weight must be 1D (one number per event) ---
    try:
        w = events.info.weights
        if ak.ndim(w) > 1:
            w = w[:, 0]  # take nominal weight
        ew = ak.to_numpy(w).astype(np.float32)
        if ew.shape[0] != n_events:
            ew = np.ones(n_events, dtype=np.float32)
    except Exception:
        ew = np.ones(n_events, dtype=np.float32)

    # --- event ids (globally unique across batches) ---
    event_ids = np.arange(event_offset, event_offset + n_events, dtype=np.int64)

    rec[event_col_fields["event_weight"].name] = pa.array(
        ew, type=event_col_fields["event_weight"].type
    )
    rec[event_col_fields["event_id"].name] = pa.array(
        event_ids, type=event_col_fields["event_id"].type
    )

    # --- jet-level lists per event ---
    for col_name, col_field in jet_col_fields.items():
        arr = getattr(jets, col_name)  # (n_events, n_jets)
        rec[col_field.name] = pa.array(arr, type=col_field.type)

    # --- constituent-level list-of-lists per event ---
    for col_name, col_field in constit_col_fields.items():
        arr = getattr(constituents, col_name)  # (n_events, n_jets, n_constit)
        rec[col_field.name] = pa.array(arr, type=col_field.type)

    for col_name, col_field in hard_parton_col_fields.items():
            arr = getattr(hard_partons, col_name)  # (n_events, n_jets, n_constit)
            rec[col_field.name] = pa.array(arr, type=col_field.type)

    return pa.RecordBatch.from_pydict(rec, schema=get_schema())


@nb.jit
def charge_from_pdgid(builder, pid_arr):
    for pids in pid_arr:
        builder.begin_list()
        for pid in pids:
            builder.append(charge(pid))
        builder.end_list()
    return builder

#  --- cuts ---
def process_jets(
    jets,
    constituents,
    jet_pt_min=5.0,
    jet_abs_rap_max=0.6,
):
    jetPtCut = jets.pt > jet_pt_min
    jetEtaCut = np.abs(jets.eta) < jet_abs_rap_max

    jets["ncharged"] = ak.count_nonzero(constituents.charge, axis=-1)
    jets["nconstituents"] = ak.count(constituents, axis=-1)
    jetNChargedCut = jets.ncharged > 1

    jetCut = jetPtCut & jetEtaCut & jetNChargedCut
    jets = ak.drop_none(ak.mask(jets, jetCut), axis=1)
    constituents = ak.drop_none(ak.mask(constituents, jetCut), axis=1)
    print(f"------After cuts, {ak.count(jets.pt)} jets left...")

    return jets, constituents


def inclusive_jets_sorted_by_pt(
    cluster_sequence: fj.ClusterSequence,
    min_pt: float = 2.0,
) -> tuple[ak.Array, ak.Array]:
    jets = cluster_sequence.inclusive_jets(min_pt=min_pt)
    sortedIndex = ak.argsort(jets.pt, axis=-1, ascending=False)
    jets = jets[sortedIndex]
    constituents = cluster_sequence.constituents(min_pt=min_pt)[sortedIndex]
    sortedConstitIndex = ak.argsort(constituents.pt, axis=-1, ascending=False)
    constituents = constituents[sortedConstitIndex]
    print(f"------Clustered {ak.num(jets, axis=0)} jets...")
    return jets, constituents


def select_particles_for_clustering(events, con_kt_min=0.2, con_abs_eta_max=1.0):
    invalids_ = ak.is_none(events.prt)
    events = events[~invalids_]

    final_state_sel_ = events.prt.status > 0
    prt_kt_sel_ = events.prt.p.pt > con_kt_min
    prt_eta_sel_ = np.abs(events.prt.p.eta) < con_abs_eta_max

    prt_sel_ = final_state_sel_ & prt_kt_sel_ & prt_eta_sel_
    events["prt"] = ak.drop_none(ak.mask(events.prt, prt_sel_), axis=1)

    n_particles_sel_ = ak.num(events.prt, axis=1) > 2
    event_mask = n_particles_sel_
    events = events[n_particles_sel_]

    return events, event_mask


def make_cluster_sequence_input(events, make_highlevel=True, **kwargs):
    events, event_mask = select_particles_for_clustering(events, **kwargs)
    if not make_highlevel:
        return events, event_mask

    coordinates_ = ["px", "py", "pz", "e"]
    coordinate_keys_ = [("prt", "p", coo) for coo in coordinates_]

    events = ak.with_field(
        events,
        charge_from_pdgid(ak.ArrayBuilder(), events.prt.id).snapshot(),
        where=("prt", "charge"),
    )

    prt_fields_ = deepcopy(events.prt.fields)
    prt_fields_.remove("p")

    coordinates_ = coordinates_ + prt_fields_
    keys_ = coordinate_keys_ + [("prt", f) for f in prt_fields_]

    candidates_ = ak.zip(
        dict(zip(coordinates_, [events[key] for key in keys_])),
        with_name="Momentum4D",
    )

    return candidates_, event_mask

def format_candidates(particles):
    #input raw data structure and format into a momentum 4vector

    candidates = ak.zip(
        {
            "px": particles.p.px,
            "py": particles.p.py,
            "pz": particles.p.pz,
            "e": particles.p.e,
            "charge": charge_from_pdgid(ak.ArrayBuilder(), particles.id).snapshot(),
            "id": particles.id,
        
        },
        with_name="Momentum4D",
    )
    return candidates

def extract_hard_partons(events):
    hard_parton = events.prt.status == -23
    hard_parton_padded = ak.drop_none(ak.mask(events.prt, hard_parton))
    hard_parton_4vector = format_candidates(hard_parton_padded)

    return hard_parton_4vector

def match_jets_to_partons(jets, partons):
    deta = jets.eta[:, :, np.newaxis] - partons.eta[:, np.newaxis, :]
    dphi = jets.phi[:, :, np.newaxis] - partons.phi[:, np.newaxis, :]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    delta_R = np.sqrt(deta**2 + dphi**2)
    indexing = ak.argmin(delta_R, axis=-1)
    min_delta_R = ak.min(delta_R, axis=-1)
    is_matched = min_delta_R < 0.3
    matched_id = partons.id[indexing]
    parton_id = ak.where(is_matched, matched_id, 0)

    return parton_id



def cluster_batch(events, jet_definition, event_offset: int = 0, con_kt_min=1.0, cs_pt_min=2.0):
    hard_partons = extract_hard_partons(events)
    events_, event_mask = make_cluster_sequence_input(events, con_kt_min=con_kt_min)
    hard_partons_filtered = hard_partons[event_mask]

    print("raw hard partons:", len(hard_partons))
    print("event mask length:", len(event_mask))
    print("filtered hard partons:", len(hard_partons_filtered))
    print("events_ length:", len(events_))

    cluster_seq_ = fj.ClusterSequence(events_, jet_definition)
    jets_, constituents_ = inclusive_jets_sorted_by_pt(cluster_seq_, min_pt=cs_pt_min)

    jets_, constituents_ = process_jets(jets_, constituents_, jet_pt_min=15.0)

    parton_id = match_jets_to_partons(jets_, hard_partons_filtered)
    jets_ = ak.with_field(jets_, parton_id, where="parton_id")

    try:
        w = events_.info.weights
        if ak.ndim(w) > 1:
            w = w[:, 0]
        jets_["weight"], _ = ak.broadcast_arrays(w, jets_.px)
    except Exception:
        jets_["weight"] = ak.ones_like(jets_.px)

    return events_to_rb_dict(events_, jets_, constituents_, hard_partons_filtered, event_offset=event_offset)


def setup_generator(seed=0) -> pythia.Pythia:
    pythia_ = pythia.Pythia()

    pythia_.readFile("pythia8_detroit_tune.txt")
    pythia_.readString("Beams:idA = 2212")
    pythia_.readString("Beams:idB = 2212")
    pythia_.readString("Beams:eCM = 200.")
    pythia_.readString("HardQCD:all = on")
    pythia_.readString("PhaseSpace:pTHatMin = 10")

    pythia_.readString("Random:setSeed = on")
    pythia_.readString(f"Random:seed = {seed}")

    return pythia_


def generate(n_events, batch_size=-1, error_mode="none", seed=0):
    pythia_ = setup_generator(seed=seed)
    pythia_.init()

    batch_size = n_events if batch_size < 0 else batch_size
    n_events_to_generate_ = n_events
    i_batch = 0

    while n_events_to_generate_ > 0:
        batch_size = (
            n_events_to_generate_ if batch_size > n_events_to_generate_ else batch_size
        )
        n_events_to_generate_ -= batch_size
        i_batch += 1
        print(f"{i_batch} batches generated...")
        yield pythia_.nextBatch(batch_size, error_mode)


def worker(n_events, batch_size, jet_definition, output_file, slot_id=None, seed=0, **kwargs):
    if slot_id is not None:
        output_file = f"{output_file}_{slot_id}.arrow"
    else:
        output_file = f"{output_file}.arrow"

    sink = pa.OSFile(output_file, "wb")
    writer = pa.ipc.RecordBatchFileWriter(sink, get_schema())

    event_offset = 0
    for batch in generate(n_events, batch_size=batch_size, seed=seed):
        rb = cluster_batch(batch, jet_definition, event_offset=event_offset)
        writer.write(rb)
        # advance by number of events written in this record batch
        event_offset += rb.num_rows

    writer.close()
    get_schema.cache_clear()


if __name__ == "__main__":
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4, fj.BIpt2_scheme)
    worker(
        1000000,
        100000,
        jet_def,
        "../datasets/Pythia_pp200GeV_Dijet_Analysis/parton_matching_con_1.0_jet_15",
    )
