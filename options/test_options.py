from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """
    def __init__(self) -> None:
        """Initialize TestOptions with testing-specific defaults and call parent constructor."""
        super().__init__()
        self.isTrain: bool = False
    
    def initialize(self, parser):
        parser = super().initialize(parser)

        parser.add_argument('--results_dir', type=str, default='results', help='saves results here.')
        parser.add_argument('--num_test', type=int, default=200, help='how many test images to run')
        parser.set_defaults(phase='test')
        parser.set_defaults(preprocess='none')
        parser.set_defaults(input_dir='input_test')
        parser.set_defaults(output_dir='output_test')
        
        return parser