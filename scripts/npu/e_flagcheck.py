"""Device-less flag check for e_crash_chain.py (runs INSIDE the saea-base image, no card, no network): does vLLM 0.30.0
parse the E2 flags? Prints one `FLAGCHECK <json>` line per candidate argv."""
import json

from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

parser = FlexibleArgumentParser()
EngineArgs.add_cli_args(parser)
for argv in (["--mm-processor-cache-type", "shm"], ["--no-enable-prefix-caching"],
             ["--mm-processor-cache-type", "bogus"]):
    row = {"argv": argv}
    try:
        args = parser.parse_args(argv)
        row.update(accepted=True, mm_processor_cache_type=args.mm_processor_cache_type,
                   mm_processor_cache_gb=args.mm_processor_cache_gb, enable_prefix_caching=args.enable_prefix_caching)
    except SystemExit as exit_:
        row.update(accepted=False, exit=str(exit_))
    print("FLAGCHECK " + json.dumps(row), flush=True)
