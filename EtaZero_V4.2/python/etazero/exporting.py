"""Call the source exporters in the controller, retaining their CLI semantics."""
from contextlib import redirect_stderr, redirect_stdout
import logging
import sys
import traceback


def execute_export(argv, stream):
    """Reuse imports across rounds; keep exporter process-local state scoped.

    Export still loads an independent CPU model from the committed checkpoint.
    The controller owns the run lock, stage timing and atomic directory publish.
    """
    handlers = logging.root.handlers[:]
    level = logging.root.level
    previous_argv = sys.argv
    scoped_handler = logging.StreamHandler(stream)
    logging.root.handlers = [scoped_handler]
    logging.root.setLevel(logging.INFO)
    try:
        with redirect_stdout(stream), redirect_stderr(stream):
            import numpy as np
            import torch

            sys.argv = list(map(str, argv[1:]))
            # load_model initializes a fresh CPU model before loading weights.
            # Its RNG use and the CLI's print/log settings must not leak out.
            with torch.random.fork_rng(devices=[]), np.printoptions():
                if sys.argv[0].endswith('/export_muzero.py'):
                    from export_muzero import main
                    main(sys.argv[1:])
                else:
                    from export_model_pytorch import main, parser
                    main(vars(parser.parse_args(sys.argv[1:])))
    except (Exception, SystemExit) as error:
        traceback.print_exc(file=stream)
        raise RuntimeError(f'Source export failed; see {stream.name}') from error
    finally:
        for handler in logging.root.handlers:
            if handler not in handlers:
                handler.close()
        scoped_handler.close()
        logging.root.handlers = handlers
        logging.root.setLevel(level)
        sys.argv = previous_argv
