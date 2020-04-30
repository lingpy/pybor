"""
Dummy command for scaffolding.
"""


def register(parser):
    parser.add_argument(
        "-n",
        default=42,
        help='dummy argument',
        type=int,
    )

def run(args):
    print("(calling dummy command) args.n -> %i" % args.n)
