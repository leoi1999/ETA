import argparse


DEFAULT_VISUAL_EMBEDDING_PATH = 'image/image.pkl'


class ConfigParser:
    @staticmethod
    def str2bool(value):
        if isinstance(value, bool):
            return value
        lowered = value.lower()
        if lowered in {'true', '1', 'yes', 'y'}:
            return True
        if lowered in {'false', '0', 'no', 'n'}:
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    @staticmethod
    def comma_separated_floats(value):
        try:
            return [float(item.strip()) for item in value.split(',') if item.strip()]
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                'Expected a comma-separated list of floating-point values.'
            ) from error

    def build_parser(self):
        parser = argparse.ArgumentParser(
            description="Train or evaluate the ETA multimodal entity-alignment model."
        )

        # Dataset selection and input sources.
        parser.add_argument(
            'dataset',
            nargs='?',
            type=str,
            default='FB_DB',
            choices=[
                'FB_DB',
                'FB_YAGO',
            ],
            help=(
                'dataset key: FB_DB/FB_YAGO use the existing MMEA datasets'
            ),
        )
        parser.add_argument('ratio', nargs='?', type=float, default=0.2, help='which seed split to use, e.g. 0.2, 0.5, 0.8')
        parser.add_argument('--data-root', type=str, default='data', help='root directory of all datasets')
        parser.add_argument('--alignment-split', choices=['auto', 'legacy', 'dmea'], default='auto', help='auto uses DMEA train/validation/test splits for pseudo-label runs')
        parser.add_argument('--split-seed', type=int, default=43, help='independent seed for the DMEA alignment split')
        parser.add_argument('--final-test', type=self.str2bool, default=True, help='evaluate held-out test once after restoring the validation-best model')
        parser.add_argument('--dataset-cache-dir', type=str, default=None, help='optional directory for reusable serialized dataset tensors')
        parser.add_argument('--embedding-cache-dir', type=str, default=None, help='optional directory for reusable dense attribute/visual embedding tables')

        # Modality selection and fusion strategy.
        parser.add_argument(
            '--modules',
            type=str,
            default='graph',
            help=(
                'comma-separated modalities to use, chosen from '
                'graph,attribute,visual'
            ),
        )
        parser.add_argument(
            '--attribute-embedding-path',
            type=str,
            default='bert_attr_from_mmkb_unified/attr_name_value_bert_cls_embeddings.pkl',
            help=(
                'path to the paper-primary entity attribute representation: '
                'serialized attribute names and values encoded by BERT CLS'
            ),
        )
        parser.add_argument('--attribute-hidden-dim', type=int, default=512, help='attribute projection-head hidden dimension')
        parser.add_argument('--attribute-projection-dim', type=int, default=256, help='attribute projection-head output dimension')
        # Defaults selected from the confirmed FB-DB fine-search winner:
        # a_temp0p07_dim256_drop0p0_lr0p001.
        parser.add_argument('--attribute-dropout', type=float, default=0.0, help='attribute projection-head dropout rate')
        parser.add_argument('--attribute-temperature', type=float, default=0.07, help='bidirectional contrastive temperature for attribute training')
        parser.add_argument('--attribute-use-projector', type=self.str2bool, default=True, help='whether to use the paper BERT-to-MLP attribute encoder; false is the frozen-BERT ablation')
        parser.add_argument(
            '--attribute-pretrained-model',
            type=str,
            default='',
            help=(
                'default pretrained attribute projector used to initialize an '
                'active trainable attribute branch; pass an empty string to train from scratch'
            ),
        )
        parser.add_argument('--visual-embedding-path', type=str, default=None, help='path to CLIP embeddings; by default uses CLIP ViT-B/16 with stability-aware layers 1,5,8, FB-DB (k=30), or FB-YAGO (k=20), using normalized votes with epsilon=1e-6')
        parser.add_argument('--visual-hidden-dim', type=int, default=512, help='optional visual projection-head hidden dimension')
        parser.add_argument('--visual-projection-dim', type=int, default=256, help='optional visual projection-head bottleneck dimension')
        parser.add_argument('--visual-dropout', type=float, default=0.1, help='optional visual projection-head dropout rate')
        parser.add_argument('--visual-temperature', type=float, default=0.07, help='contrastive temperature for optional visual projection training')
        parser.add_argument('--visual-use-projector', type=self.str2bool, default=True, help='train a shallow projector over frozen stability-reweighted visual embeddings')
        parser.add_argument('--decision-fusion-hidden-dim', type=int, default=32, help='hidden dimension of the pair-specific Dynamic Decision Fusion gating network')
        parser.add_argument('--decision-fusion-temperature', type=float, default=1.0, help='softmax temperature used to produce pair-specific modality weights')
        parser.add_argument('--decision-fusion-initial-weights', type=float, nargs=3, default=None, metavar=('GRAPH', 'ATTRIBUTE', 'VISUAL'), help='global graph/attribute/visual weights used to initialize Dynamic Decision Fusion; defaults to 0.60 0.25 0.15 when no fusion checkpoint is loaded')
        parser.add_argument('--decision-score-temperature', type=float, default=0.05, help='temperature used by the supervised fused decision-score objective')
        parser.add_argument('--decision-fusion-loss-weight', type=float, default=1.0, help='weight of the supervised Dynamic Decision Fusion loss')
        parser.add_argument('--decision-fusion-balance-weight', type=float, default=0.02, help='weight of the anti-collapse regularizer for pair-specific modality weights')
        parser.add_argument('--decision-fusion-anchor-weight', type=float, default=0.0, help='weight of the L2 penalty anchoring a continued Dynamic Decision Fusion run to its loaded initialization')
        parser.add_argument('--decision-fusion-chunk-size', type=int, default=512, help='row chunk size used for full candidate-matrix Dynamic Decision Fusion')
        parser.add_argument('--decision-fusion-detach-scores', type=self.str2bool, default=True, help='detach modality-specific scores so the fusion objective trains only Dynamic Decision Fusion')
        parser.add_argument('--train-decision-fusion-only', type=self.str2bool, default=False, help='freeze all modality encoders and train only Dynamic Decision Fusion')
        parser.add_argument('--frozen-evidence-cache', type=str, default='', help='optional reusable cache file for frozen graph/attribute/visual entity representations')

        # Pseudo labeling control.
        parser.add_argument('--semi-prediction', dest='semi_prediction', type=self.str2bool, default=False, help='enable pseudo pair generation')
        parser.add_argument('--conflict-policy', type=int, default=2, help='pseudo-pair conflict solving policy')
        parser.add_argument('--threshold', type=float, default=0.95, help='pseudo-pair acceptance threshold')
        parser.add_argument('--threshold-weight', type=float, default=0.95, help='threshold decay rate after each pseudo step')
        parser.add_argument('--pseudo-pool-path', type=str, default='', help='optional persistent shared pseudo-label pool')
        parser.add_argument('--max-pseudo-pairs-per-round', type=int, default=0, help='optional cap on newly accepted pseudo pairs per round; 0 means unlimited')

        # Model architecture and optimization.
        parser.add_argument(
            '--device',
            type=str,
            default='auto',
            help='torch device (auto, cpu, cuda, or cuda:N); auto selects cuda:0 when available',
        )
        parser.add_argument('--epochs', type=int, default=70, help='number of training epochs')
        parser.add_argument('--batch-size', type=int, default=512, help='training batch size')
        parser.add_argument('--balanced-batches', type=self.str2bool, default=True, help='rebalance each epoch into nearly equal batches and avoid a tiny final contrastive batch')
        parser.add_argument('--lr', type=float, default=None, help='optimizer learning rate; omitted selects 0.005 for graph, 0.001 for attribute, 0.0003 for visual, and 0.0001 for fusion-only training')
        parser.add_argument('--weight-decay', type=float, default=0.0, help='AdamW weight decay')
        parser.add_argument('--lr-gamma', type=float, default=0.999, help='gamma for exponential learning-rate decay')
        parser.add_argument('--scheduler-type', type=str, default='exponential', choices=['exponential', 'warmup_cosine'], help='learning-rate scheduler; warmup_cosine activates warmup-ratio and min-lr-ratio')
        parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
        parser.add_argument('--shuffle', type=self.str2bool, default=True, help='shuffle training pairs each epoch')
        parser.add_argument('--depth', type=int, default=3, help='encoder depth')
        parser.add_argument('--node-dim', type=int, default=128, help='node feature dimension')
        parser.add_argument('--relation-dim', type=int, default=128, help='relation feature dimension')
        parser.add_argument('--message-filter-hidden-dim', type=int, default=96, help='hidden dimension of the dimension-wise message-filtering MLP')
        parser.add_argument('--message-filter-input', type=str, default='embeddings', choices=['embeddings', 'input_features', 'constant', 'none'], help='node state used to compute dimension-wise message-filter weights; constant uses message-filter-scale as a fixed message coefficient')
        parser.add_argument('--message-filter-scale', type=float, default=2.0, help='message-filtering scaling coefficient eta')
        parser.add_argument('--use-relation-reflection', type=self.str2bool, default=True, help='whether to reflect neighbor messages across the relation hyperplane')
        parser.add_argument('--layer-weight-mode', type=str, default='learned', choices=['learned', 'uniform', 'fixed', 'none'], help='adaptive propagation-layer weighting mode')
        parser.add_argument('--layer-fixed-weights', type=self.comma_separated_floats, default=None, help='comma-separated depth+1 propagation-layer weights used in fixed mode')
        parser.add_argument('--layer-regularization-weight', type=float, default=5.0, help='lambda_reg for KL regularization of the learned layer-importance distribution')
        parser.add_argument('--layer-prior-type', type=str, default='folded_normal', choices=['uniform', 'folded_normal', 'poisson'], help='prior distribution for adaptive layer weights; folded_normal matches paper Eq. (8)')
        parser.add_argument('--layer-prior-loc', type=float, default=1.0, help='folded-normal prior location; negative uses the current graph depth')
        parser.add_argument('--layer-prior-scale', type=float, default=1.0, help='folded-normal prior scale for propagation layers')
        parser.add_argument('--layer-prior-rate', type=float, default=-1.0, help='Poisson prior rate; negative uses the current graph depth')
        parser.add_argument(
            '--hard-negative-beta',
            type=float,
            default=30.0,
            help='hard-negative aggregation coefficient beta in the graph alignment loss',
        )
        parser.add_argument('--warmup-ratio', type=float, default=0.1, help='fraction of total update steps used for linear warmup')
        parser.add_argument('--min-lr-ratio', type=float, default=0.01, help='final learning rate ratio for cosine decay')
        parser.add_argument('--seed', type=int, default=1029, help='Python, NumPy, and PyTorch random seed')

        # Evaluation and auxiliary analysis switches.
        parser.add_argument('--eval-interval', type=int, default=10, help='evaluate every N epochs')
        parser.add_argument('--decision-fusion-eval-interval', type=int, default=5, help='evaluation interval used specifically for Dynamic Decision Fusion training')
        parser.add_argument('--eval-start-epoch', type=int, default=0, help='start evaluation from this epoch index')
        parser.add_argument('--eval-normalization-temperature', type=float, default=0.01, help='temperature of row/column log-space score normalization')
        parser.add_argument('--eval-normalization-rounds', type=int, default=30, help='number of alternating row/column normalization rounds')
        parser.add_argument(
            '--early-stop-patience',
            type=int,
            default=0,
            help=(
                'stop after this many consecutive evaluations without a '
                'strict Hits@1 improvement; zero disables early stopping'
            ),
        )
        parser.add_argument('--semi-start-epoch', type=int, default=0, help='start pseudo labeling from this epoch index')
        parser.add_argument('--export-dev-predictions', type=str, default=None, help='optional JSON path for per-dev-pair top-1 evaluation predictions')
        parser.add_argument('--eval-only', type=self.str2bool, default=False, help='skip training and run one evaluation pass')
        parser.add_argument('--load-graph-model', type=str, default='', help='path to a saved graph-modality checkpoint for initialization or evaluation')
        parser.add_argument('--load-attribute-model', type=str, default=None, help='path to a saved attribute encoder checkpoint for initialization or evaluation')
        parser.add_argument('--load-visual-model', type=str, default='', help='path to a saved visual encoder/projector checkpoint for initialization or evaluation')
        parser.add_argument('--load-decision-fusion-model', type=str, default='', help='path to a saved Dynamic Decision Fusion checkpoint for initialization or evaluation')
        parser.add_argument('--export-graph-negatives', type=self.str2bool, default=False, help='export graph-only wrong top-1 dev predictions as negative examples')
        parser.add_argument('--graph-negatives-dir', type=str, default=None, help='directory for exported graph-only negative examples')

        # Logging output.
        parser.add_argument('--log-root', type=str, default='logs', help='root directory for training log folders; default is <repo>/logs')
        parser.add_argument('--log-name', type=str, default=None, help='training log folder name; empty uses dataset-modality-timestamp')

        return parser

    def parse_args(self):
        parser = self.build_parser()
        args = parser.parse_args()
        modules = []
        for raw_module in args.modules.split(','):
            module = raw_module.strip().lower()
            if module not in {'graph', 'attribute', 'visual'}:
                parser.error("unsupported module '{}'; choose graph, attribute, or visual".format(module))
            if module and module not in modules:
                modules.append(module)
        args.modules = ','.join(modules)
        if args.visual_embedding_path is None:
            args.visual_embedding_path = DEFAULT_VISUAL_EMBEDDING_PATH
        if (
            args.decision_fusion_initial_weights is None
            and not args.load_decision_fusion_model
        ):
            args.decision_fusion_initial_weights = [0.60, 0.25, 0.15]
        if args.lr is None:
            if args.train_decision_fusion_only:
                args.lr = 1e-4
            elif len(modules) == 1 and modules[0] == 'attribute':
                args.lr = 1e-3
            elif len(modules) == 1 and modules[0] == 'visual':
                args.lr = 3e-4
            else:
                args.lr = 5e-3
        return args
