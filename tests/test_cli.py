from ashare_quant.cli import _build_parser


def test_sync_cli_defaults_to_csi800_and_isolated_directory():
    args = _build_parser().parse_args(["sync-data"])

    assert args.universe == "csi800"
    assert args.data_dir == "data/csi800_market"
