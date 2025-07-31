from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--print_freq', type=int, default=100, help='console print frequency')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='save latest model every N iterations')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='save checkpoint every N epochs')
        parser.add_argument('--save_by_iter', action='store_true')
        parser.add_argument('--continue_train', action='store_true', help='resume training')
        parser.add_argument('--epoch_count', type=int, default=1, help='start counting epochs from here')
        parser.add_argument('--n_epochs', type=int, default=100, help='epochs with initial LR')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='epochs to linearly decay LR to 0')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term for Adam')
        parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self):
        super().__init__()
        self.isTrain = True
