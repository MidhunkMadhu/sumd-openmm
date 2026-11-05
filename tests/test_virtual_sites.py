"""GROMACS virtual sites: reading them for selections, and the molecules they belong to."""

import pytest

parmed = pytest.importorskip("parmed")

from sumd_openmm import selection
from sumd_openmm.config import SumdConfig
from sumd_openmm.selection import _molecule_ids, load_structure, resolve_one, virtual_site_links

HEAD = """[ defaults ]
1 2 yes 1.0 1.0
[ atomtypes ]
CG 6 12.011 0.0 A 0.35 0.3
BR 35 79.904 0.0 A 0.35 0.3
LP 0 0.0 0.0 A 0.0 0.0
OW 8 15.999 0.0 A 0.31 0.6
HW 1 1.008 0.0 A 0.0 0.0
[ bondtypes ]
CG CG 1 0.15 1000
CG BR 1 0.19 1000
OW HW 1 0.1 1000
[ angletypes ]
HW OW HW 1 104.5 300
"""
LIGAND = """[ moleculetype ]
LIG 3
[ atoms ]
1 CG 1 LIG C1 1 0.0 12.011
2 CG 1 LIG C2 1 0.0 12.011
3 BR 1 LIG BR1 1 -0.1 79.904
4 LP 1 LIG LP1 1 0.1 0.0
[ bonds ]
1 2 1
2 3 1
"""
WATER = """[ moleculetype ]
SOL 2
[ atoms ]
1 OW 1 SOL OW 1 -0.8 15.999
2 HW 1 SOL HW1 1 0.4 1.008
3 HW 1 SOL HW2 1 0.4 1.008
[ bonds ]
1 2 1
1 3 1
[ angles ]
2 1 3 1
"""
TAIL = "[ system ]\nt\n[ molecules ]\n"


def topology(tmp_path, vsite, molecules="LIG 2\n", extra=""):
    path = tmp_path / "t.top"
    path.write_text(HEAD + WATER + LIGAND + extra + vsite + TAIL + molecules)
    return str(path)


def molecules_of(path, log=print):
    struct = load_structure(path, log)
    return len(struct.atoms), len(set(_molecule_ids(struct, virtual_site_links(path))))


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("vsite", [
    "[ virtual_sites2 ]\n4 3 2 1 0.5\n",
    "[ virtual_sites3 ]\n4 3 2 1 3 0.1 0.1\n",              # function type parmed lacks
    "[ virtual_sites3 ]\n4 3 2 1 1 0.1 0.2\n",              # different weights
    "[ virtual_sites3 ]\n4 3 2 1 1 0.1 0.1\n",              # chain frame: ValueError in parmed
    "[ virtual_sites4 ]\n4 3 2 1 1 2 0.1 0.1 0.1\n",
    "[ virtual_sitesn ]\n4 1 3 2\n",
    "[ dummies2 ]\n4 3 2 1 0.5\n",
])
def test_every_site_stays_in_its_molecule(tmp_path, vsite):
    assert molecules_of(topology(tmp_path, vsite), lambda m: None) == (8, 2)


@pytest.mark.filterwarnings("ignore")
def test_missing_bond_parameters_fall_back(tmp_path):
    # a KeyError in parmed's geometry step: the bond type comes after the molecule
    text = LIGAND.replace("2 3 1\n", "2 3 1\n1 3 1\n")
    path = tmp_path / "t.top"
    path.write_text(HEAD.replace("CG BR 1 0.19 1000\n", "") + WATER + text +
                    "[ virtual_sites3 ]\n4 3 2 1 1 0.1 0.1\n[ bondtypes ]\nCG BR 1 0.19 1000\n" + TAIL + "LIG 1\n")
    assert molecules_of(str(path), lambda m: None) == (4, 1)


@pytest.mark.filterwarnings("ignore")
def test_site_bonded_to_its_parent_is_still_an_error(tmp_path):
    path = topology(tmp_path, "[ virtual_sites3 ]\n4 3 2 1 3 0.1 0.1\n", extra="[ bonds ]\n3 4 1 0.1 1000\n")
    with pytest.raises(parmed.exceptions.GromacsError):
        load_structure(path, lambda m: None)


@pytest.mark.filterwarnings("ignore")
def test_links_count_copies_and_earlier_molecules(tmp_path):
    path = topology(tmp_path, "[ virtual_sites2 ]\n4 3 2 1 0.5\n", molecules="SOL 2\nLIG 2\nSOL 1\n")
    assert virtual_site_links(path) == [(9, 8), (13, 12)]
    assert molecules_of(path) == (17, 5)


@pytest.mark.filterwarnings("ignore")
def test_links_follow_the_preprocessor(tmp_path):
    vsite = "#ifdef NOT_DEFINED\n[ virtual_sites2 ]\n4 3 2 1 0.5\n#endif\n"
    assert virtual_site_links(topology(tmp_path, vsite)) == []


@pytest.mark.filterwarnings("ignore")
def test_one_warning_through_the_log_and_the_site_is_selected(tmp_path):
    messages = []
    path = topology(tmp_path, "[ virtual_sites3 ]\n4 3 2 1 3 0.1 0.1\n")
    struct = load_structure(path, messages.append)
    assert len(messages) == 1 and "1 virtual site definition(s)" in messages[0] and "&!@LP1" in messages[0]
    assert list(resolve_one(struct, ":1&!@H=")[0]) == [0, 1, 2, 3]
    assert list(resolve_one(struct, ":1&!@H=&!@LP1")[0]) == [0, 1, 2]


@pytest.mark.filterwarnings("ignore")
def test_supported_site_keeps_parmeds_geometry_and_gives_no_warning(tmp_path):
    messages = []
    water = WATER.replace("SOL", "TP4").replace("3 HW 1 TP4 HW2 1 0.4 1.008\n",
                                               "3 HW 1 TP4 HW2 1 0.4 1.008\n4 LP 1 TP4 MW 1 0.0 0.0\n")
    path = tmp_path / "t.top"
    path.write_text(HEAD + water + "[ virtual_sites3 ]\n4 1 2 3 1 0.128 0.128\n" + TAIL + "TP4 1\n")
    struct = load_structure(str(path), messages.append)
    assert not messages
    site = [b for b in struct.bonds if b.atom1.name == "MW" or b.atom2.name == "MW"]
    assert len(site) == 1 and site[0].type.req > 0


def test_parser_is_restored_even_after_an_error(tmp_path):
    from parmed.gromacs.gromacstop import GromacsTopologyFile
    original = GromacsTopologyFile._parse_vsites3
    with pytest.raises(Exception):
        load_structure(str(tmp_path / "missing.top"))
    assert GromacsTopologyFile._parse_vsites3 is original


def test_missing_file_is_read_once_even_with_vsite_in_its_path(tmp_path, monkeypatch):
    calls = []
    real = parmed.load_file

    def spy(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(parmed, "load_file", spy)
    with pytest.raises(FileNotFoundError):
        load_structure(str(tmp_path / "vsite_tests" / "a.top"))
    assert len(calls) == 1


def test_a_parmed_without_the_parser_still_reads_topologies(tmp_path, monkeypatch):
    from parmed.gromacs.gromacstop import GromacsTopologyFile
    monkeypatch.delattr(GromacsTopologyFile, "_parse_vsites3")
    path = tmp_path / "a.pdb"
    path.write_text("ATOM      1  C1  MOL A   1       5.000   5.000   5.000  1.00  0.00           C\nEND\n")
    assert len(load_structure(str(path)).atoms) == 1


@pytest.mark.filterwarnings("ignore")
def test_resolve_keeps_a_ligand_with_its_site_in_one_component(tmp_path):
    path = topology(tmp_path, "[ virtual_sites2 ]\n4 3 2 1 0.5\n", molecules="LIG 1\n")
    config = SumdConfig.from_cfg(dict(metric_1_type="distance", metric_1_a="indices:0-3",
                                      metric_1_b="indices:0", metric_1_role="supervise",
                                      metric_1_direction="decrease", metric_1_target="1"))
    evaluator, _ = selection.resolve(config, path)
    assert len(evaluator.metrics[0].components["a"]) == 1
