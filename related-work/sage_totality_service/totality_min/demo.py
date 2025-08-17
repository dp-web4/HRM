
# demo.py
from totality_core import TotalityStore
from transforms import context_shift, semantic_variation
import json

def main():
    T = TotalityStore()

    # Seed two simple schemes
    grasp = T.add_scheme("grasp(object)", slots=[{"name":"object","value":"cube"}], activation=0.6)
    place = T.add_scheme("place(object,location)", slots=[{"name":"object","value":"cube"},{"name":"location","value":"table"}], activation=0.4)

    # Create a base canvas combining them
    base = T.add_canvas(description="base scene", entities=[grasp, place])

    print("=== READ (activation>=0.0) ===")
    print(json.dumps(T.to_jsonable(), indent=2))

    # Imagine: produce 2 canvases with naive value-permutation op
    canvases = T.imagine_from_schemes([grasp.id, place.id], ops=[{"op":"value_perm"}], count=2)
    print("\n=== IMAGINE (2 canvases) ===")
    print(json.dumps([{"id":c.id, "desc":c.description, "ops":c.provenance["ops"]} for c in canvases], indent=2))

    # Activate: upweight grasp
    T.activate([{"id": grasp.id, "delta": 0.2}])
    print("\n=== ACTIVATE (grasp +0.2) ===")
    print(json.dumps({"grasp_activation": T.activations[grasp.id]}, indent=2))

    # Write: commit a distilled abstraction
    abstraction = grasp.clone()
    abstraction.label = "manipulate(item)"
    for sl in abstraction.slots:
        sl.value = semantic_variation(str(sl.value))
    committed = T.write_schemes([abstraction], mode="merge")
    print("\n=== WRITE (distilled abstraction) ===")
    print(json.dumps({"committed": committed}, indent=2))

    # Show final store
    print("\n=== FINAL STORE ===")
    print(json.dumps(T.to_jsonable(), indent=2))

if __name__ == "__main__":
    main()
