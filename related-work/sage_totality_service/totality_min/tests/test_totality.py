
# tests/test_totality.py
from totality_core import TotalityStore

def test_seed_and_read():
    T = TotalityStore()
    s = T.add_scheme("grasp(object)", slots=[{"name":"object","value":"cube"}], activation=0.5)
    assert s.id in T.schemes
    data = T.read(activation_min=0.4)
    assert any(x.id == s.id for x in data["schemes"])

def test_imagine_and_write():
    T = TotalityStore()
    a = T.add_scheme("place(object,location)", slots=[{"name":"object","value":"cube"},{"name":"location","value":"table"}], activation=0.3)
    canv = T.imagine_from_schemes([a.id], ops=[{"op":"value_perm"}], count=1)
    assert len(canv) == 1
    abstr = a.clone()
    abstr.label = "manipulate(item)"
    ids = T.write_schemes([abstr], mode="merge")
    assert ids[0] in T.schemes
