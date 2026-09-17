import json
from pathlib import Path

import pytest

from rona.cli import build_parser, main

SEQ = "GGCGCGGCACCGUCCGCGGAACAAACGG"


def test_parser_builds():
    parser = build_parser()
    args = parser.parse_args(["fold", SEQ, "-n", "3", "--rate", "50"])
    assert args.trajectories == 3 and args.rate == 50.0


def test_pause_argument_parsing():
    parser = build_parser()
    args = parser.parse_args(["fold", SEQ, "--pause", "20:1.5", "--pause", "30:2:s"])
    assert [(p.position, p.duration, p.stochastic) for p in args.pause] == [
        (20, 1.5, False),
        (30, 2.0, True),
    ]
    with pytest.raises(SystemExit):
        parser.parse_args(["fold", SEQ, "--pause", "nonsense"])


def test_info_command(capsys):
    assert main(["info", SEQ]) == 0
    out = capsys.readouterr().out
    assert "stems" in out and "sites" in out


def test_trajectory_command(capsys):
    assert main(["trajectory", SEQ, "--frames", "8", "--post-time", "0.3", "--k-zip", "3e5"]) == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if not l.startswith("#")]
    assert len(lines) == 8


def test_fold_writes_requested_outputs(tmp_path, capsys):
    code = main([
        "fold", SEQ, "-n", "4", "-o", str(tmp_path), "--prefix", "t",
        "--frames", "8", "--post-time", "0.3", "--rate", "60",
        "--k-zip", "3e5",
        "--json", "--svg", "--html", "--workers", "1", "-q",
    ])
    assert code == 0
    assert (tmp_path / "t.json").exists()
    assert (tmp_path / "t.html").exists()
    for label in ("occupancy", "energy", "pairprob", "pseudoknots", "final"):
        assert (tmp_path / f"t.{label}.svg").exists(), label
    payload = json.loads((tmp_path / "t.json").read_text())
    assert payload["sequence"] == SEQ
    assert payload["n_trajectories"] == 4


def test_fold_reads_fasta(tmp_path):
    fasta = tmp_path / "s.fa"
    fasta.write_text(f">demo\n{SEQ}\n")
    code = main([
        "fold", str(fasta), "-n", "2", "-o", str(tmp_path), "--frames", "5",
        "--post-time", "0.3", "--k-zip", "3e5", "--json", "--workers", "1", "-q",
    ])
    assert code == 0
    assert (tmp_path / "demo.json").exists()


def test_fasta_description_does_not_leak_into_filenames(tmp_path):
    """A record name is the first header token; output paths stay safe."""
    fasta = tmp_path / "s.fa"
    fasta.write_text(f">my_rna  a long description: with punctuation/slashes\n{SEQ}\n")
    code = main([
        "fold", str(fasta), "-n", "2", "-o", str(tmp_path), "--frames", "5",
        "--post-time", "0.3", "--k-zip", "3e5", "--json", "--workers", "1", "-q",
    ])
    assert code == 0
    assert (tmp_path / "my_rna.json").exists()


def test_prefix_is_sanitised(tmp_path):
    code = main([
        "fold", SEQ, "-n", "2", "-o", str(tmp_path), "--prefix", "a b/c:d",
        "--frames", "5", "--post-time", "0.3", "--k-zip", "3e5", "--json", "--workers", "1", "-q",
    ])
    assert code == 0
    assert (tmp_path / "a_b_c_d.json").exists()


def test_no_pseudoknots_flag_forbids_crossings(tmp_path):
    main([
        "fold", SEQ, "-n", "6", "-o", str(tmp_path), "--prefix", "np",
        "--frames", "6", "--post-time", "0.3", "--k-zip", "3e5", "--no-pseudoknots",
        "--json", "--workers", "1", "-q",
    ])
    payload = json.loads((tmp_path / "np.json").read_text())
    assert all(v == 0.0 for v in payload["pseudoknot_fraction"])
    assert all("[" not in s for s in payload["structures"])
