#############################################
## Artemis                                 ##
## Copyright (c) 2022-present NAVER Corp.  ##
## CC BY-NC-SA 4.0                         ##
#############################################
import os
import argparse
from config import MAIN_DIR, VOCAB_DIR, CKPT_DIR, RANKING_DIR, HEATMAP_DIR

print("Main directory (default root for vocabulary files, model checkpoints, ranking files, heatmaps...): "+MAIN_DIR)

parser = argparse.ArgumentParser()

parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--learning_rate', type=float, default=0.0000025, help='init learning rate')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
# Data paths & directories
parser.add_argument('--data_name', default='fashionIQ', choices=('fashionIQ', 'fashion200K', 'shoes', 'cirr', 'amazon'), help='Dataset name (fashionIQ|fashion200K|shoes|cirr).')
parser.add_argument('--vocab_dir', default=VOCAB_DIR, help='Path to saved vocabulary pickle files')

# Directories for experiment outputs (logging, ckpt, rankings/predictions)
parser.add_argument('--exp_name', default='X', help='Experiment name, used as sub-directory to save experiment-related files (model, ranking files, heatmaps...).')
parser.add_argument('--ckpt_dir', default=CKPT_DIR, help='Directory in which to save the models from the different experiments.')
parser.add_argument('--ranking_dir', default=RANKING_DIR, type=str, help='Directory in which to save the ranking/prediction files, if any to save.')
parser.add_argument('--heatmap_dir', default=HEATMAP_DIR, type=str, help='Directory in which to save the heatmaps.')

# Data/Dataloaders parameters
parser.add_argument('--batch_size', default=32, type=int, help='Size of a mini-batch.')
parser.add_argument('--crop_size', default=224, type=int, help='Size of an image crop as the CNN input.')
parser.add_argument('--workers', default=0, type=int, help='Number of data loader workers.')
parser.add_argument('--categories', default='all', type=str, help='Names of the data categories to consider for a given dataset. Category names must be separated with a space. Specify "all" to consider them all (the interpretation of "all" depends on the dataset).')

# Model settings
parser.add_argument('--model_version', default='ARTEMIS', choices=('cross-modal', 'visual-search', 'late-fusion', 'EM-only', 'IS-only', 'ARTEMIS', "TIRG"), help='Model version (ARTEMIS, one of ARTEMIS ablatives, or TIRG)')
parser.add_argument('--ckpt', default='None', type=str, metavar='PATH', help='Path of the ckpt to resume from (default: none).')
parser.add_argument('--embed_dim', default=512, type=int, help='Dimensionality of the final text & image embeddings.')
parser.add_argument('--cnn_type', default='resnet50', help='The CNN used as image encoder.')
parser.add_argument('--load_image_feature', default=0, type=int, help="Whether (if int > 0) to load pretrained image features instead of loading raw images and using a cnn backbone. Indicate the size of the feature (int).")
parser.add_argument('--txt_enc_type', default='bigru', choices=('bigru', 'lstm'), help="The text encoder (bigru|lstm).")
parser.add_argument('--lstm_hidden_dim', default=1024, type=int, help='Number of hidden units in the LSTM.')
parser.add_argument('--wemb_type', default='glove', choices=('glove', 'None'), type=str, help='Word embedding (glove|None).')
parser.add_argument('--word_dim', default=300, type=int, help='Dimensionality of the word embedding.')
parser.add_argument('--use_clip',type=bool,default=False)
parser.add_argument('--clip_img_encoder_type',type=str,default='RN50')
parser.add_argument('--LEARNING_RATE',type=float,default=0.0001)
parser.add_argument('--PRETRAINED_WEIGHT_LR_FACTOR_IMAGE',type=float,default=0.1)
parser.add_argument('--WEIGHT_DECAY',type=float,default=1e-6)
parser.add_argument('--PRETRAINED_WEIGHT_LR_FACTOR_TEXT',type=int,default=1)
# Training / optimizer settings
parser.add_argument('--num_epochs', default=50, type=int, help='Number of training epochs.')
parser.add_argument('--lr', default=.0005, type=float, help='Initial learning rate.')
parser.add_argument('--step_lr', default=10, type=int, help="Step size, number of epochs after which to apply a learning rate decay.")
parser.add_argument('--gamma_lr', default=0.5, type=float, help="Learning rate decay.")
parser.add_argument('--learn_temperature', action='store_true', help='Whether to use and learn the temperature parameter (the scores given to the loss criterion are multiplied by a trainable version of --temperature).')
parser.add_argument('--temperature', default=2.65926, type=float, help='Temperature parameter. 2.65926')
parser.add_argument('--validate', default='val', choices=('val', 'test', 'test-val','dev'), help='Split(s) on which the model should be validated (if 2 are given, 2 different checkpoints of model_best will be kept, one for each validating split).')
parser.add_argument('--log_step', default=50, type=int, help='Every number of steps the log will be printed.')
parser.add_argument('--img_finetune', action='store_true', help='Fine-tune CNN image encoder.')
parser.add_argument('--txt_finetune', action='store_true', help='Fine-tune the word embeddings.')

# Other t1:lstm+step_lr = 20    t:bigru+step_lr=10
parser.add_argument('--gradcam', action='store_true', help='Keep gradients & activations computed while encoding the images to further interprete what the network uses to make its decision.')
parser.add_argument('--studied_split', default="val", help="Split to be used for the computation (this does not impact the usual training & validation pipeline, but impacts other scripts (for evaluation or visualizations purposes)).")

# experiment directory
parser.add_argument('--save', type=str, default='EXP', help='where to save the experiment')
parser.add_argument("--drpt", action="store", default=0.1, dest="drpt", type=float, help="dropout")



#  search

parser.add_argument('--warm_up_epochs',type=int,default=20)
parser.add_argument('--reg_coefficient',type=int,default=1)
parser.add_argument('--time_window',type=int,default=2)
parser.add_argument('--perturb_alpha', type=str, default='random', help='perturb for alpha')
parser.add_argument('--epsilon_alpha', type=float, default=0.3, help='max epsilon for alpha')

# number of input features
parser.add_argument('--num_input_nodes', type=int, help='cell input', default=2)
parser.add_argument('--num_keep_edges', type=int, help='cell step connect', default=2)

# for cells and steps and inner representation size
parser.add_argument('--multiplier', type=int, help='cell output concat', default=2)
parser.add_argument('--cell_num', type=int, help='cell steps', default=3)
parser.add_argument('--node_steps', type=int, help='inner node steps', default=1)
#
# archtecture optimizer
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--status',type=str,default='search')
parser.add_argument('--parallel', help='Use several GPUs', action='store_true', dest='parallel',
                        default=False)

# parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

# MAAF settings
parser.add_argument('--maaf_num_heads',type=int,default=2)
parser.add_argument('--ff_width',type=int,default=256)
parser.add_argument('--dropout',type=float,default=0.1)
parser.add_argument('--num_blocks',type=int,default=1)
parser.add_argument('--position_encodings',type=str,default="sinusoidal",help='mixed')
parser.add_argument('--output',type=str,default="rwpool")

# rtic
parser.add_argument('--n_blocks',type=int,default=1)
parser.add_argument('--act_fn',type=str,default='LeakyReLU')

parser.add_argument('--alpha_scale',type=int,default=1)
parser.add_argument('--beta_scale',type=int,default=1)

parser.add_argument('--loss_type',type=str,default="ce_loss")

parser.add_argument('--fusion',type=str,default="concat")
def verify_input_args(args):
	"""
	Check that saving directories exist (or create them).
	Define default values for each dataset.
	"""

	############################################################################
	# --- Check that directories exist (or create them)

	# training ckpt directory
	ckpt_dir = os.path.join(args.ckpt_dir, args.exp_name)
	if not os.path.isdir(args.ckpt_dir):
		print('Creating a directory: {}'.format(ckpt_dir))
		os.makedirs(ckpt_dir)

	# validated ckpt directories
	args.validate = args.validate.split('-') # splits for validation
	for split in args.validate:
		ckpt_val_dir = os.path.join(ckpt_dir, split)
		if not os.path.isdir(ckpt_val_dir):
			print('Creating a directory: {}'.format(ckpt_val_dir))
			os.makedirs(ckpt_val_dir)

	# prediction directory
	if not os.path.isdir(args.ranking_dir):
		os.makedirs(args.ranking_dir)

	############################################################################
	# --- Process input arguments: deduce some new arguments from provided ones.

	if args.wemb_type == "None":
		args.wemb_type = None

	# Number and name of data categories
	args.name_categories = [None]
	args.recall_k_values = [1, 10, 50]
	args.recall_subset_k_values = None
	args.study_per_category = False # to evaluate the model on each category separately (case of a dataset with multiple categories)
	if args.data_name == 'fashionIQ':
		args.name_categories = ["dress", "shirt", "toptee"]
		args.recall_k_values = [10, 50]
		args.study_per_category = True
	elif args.data_name == 'cirr':
		args.recall_k_values = [1, 5, 10, 50]
		args.recall_subset_k_values = [1, 2, 3]
	args.number_categories = len(args.name_categories)
		
	return args