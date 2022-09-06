import argparse
import time
from os import path
from os import path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import evaluate
import utils
from models import ProtoNetItemUserLL

parser = argparse.ArgumentParser(description='Prototypical Network for Recommendation')
parser.add_argument('--dataset', type=str, default='epinions0.3')
parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.000, help='weight decay coefficient')
parser.add_argument('--epochs', type=int, default=400, help='upper epoch limit')
parser.add_argument('--embed_dim_item', type=int, default=128, help='embedding size of item')
parser.add_argument('--embed_dim_user', type=int, default=128,
                    help='embedding size of user, this is only used in ProtoNetUserItemLL')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_false', default=True, help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
parser.add_argument('--classes_per_it', type=int, default=512, help='number of random classes per episode for training')
parser.add_argument('--num_samples', type=int, default=5,
                    help='number of samples per class to use as query for training')
parser.add_argument('--item_class', action='store_true', help='Use item as classes')
parser.add_argument('--n_memory', type=int, default=100, help="number of memory vectors")
parser.add_argument('--model_variant', type=str, default='memory', choices=['gnn', 'memory', 'pronet'])
parser.add_argument('--base_model', type=str, default='vae', choices=['bpr', 'vae', 'dae'],
                    help="choice of base model")
parser.add_argument('--loss_type', default='multi', choices=['bce', 'multi'], help="choice of loss function model")
parser.add_argument('--l2_lambda', type=float, default=0.005, help="L2 loss weight")
parser.add_argument('--base_training', action='store_true', default=False, help='Force to train base model')

# TODO @Aravind: meta-learning baselines: LWA, NLBA, MAML, ala-carte.

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
np.random.seed(args.seed)
# base_method = 'bpr'

added = '_new_nolambda123'
save = "/data/model/our_model/{}_{}_{}_{}_{}_{}{}.pt".format(args.base_model, args.model_variant, args.dataset,
                                                     args.num_samples, args.l2_lambda, args.classes_per_it, added)
np_save = args.dataset + "_" + args.model_variant + "_" + str(args.num_samples) + "_" + str(args.l2_lambda) + "_" + str(args.classes_per_it) + added

base_method = args.base_model
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###############################################################################
# Load data
###############################################################################

# Define train/val/test datasets
st = time.time()

train_dataset = utils.TrainDatasetProNet(args.dataset)
train_df = train_dataset.train_data_ui

val_dataset_popular = utils.EvalDatasetPopularProNet(args.dataset, train_dataset.train_data_ui_popular, datatype='val')
test_dataset_popular = utils.EvalDatasetPopularProNet(args.dataset, train_dataset.train_data_ui_popular, datatype='test')

val_dataset_few_shot = utils.EvalDatasetFewShotProNet(args.dataset, train_dataset.n_users,
                                                      datatype='val',
                                                      train_data_ui=train_dataset.train_data_ui)

test_dataset_few_shot = utils.EvalDatasetFewShotProNet(args.dataset, train_dataset.n_users,
                                                       datatype='test',
                                                       train_data_ui=train_dataset.train_data_ui)
print("data loading time : ", time.time() - st)
val_query_users = val_dataset_few_shot.query_users
test_query_users = test_dataset_few_shot.query_users

n_items = train_dataset.n_items
n_users = train_dataset.n_users

# Training: Batch size is based on # items.
params = {'batch_size': args.classes_per_it,
          'num_workers': 8, 'pin_memory': True}

# Evaluation: Batch-size is based on # items.
eval_params = {'shuffle': False, 'batch_size': 32,
               'num_workers': 8, 'pin_memory': True}

# Define data loaders.
train_loader = torch.utils.data.DataLoader(train_dataset, **params)
if base_method == 'vae' or base_method == 'dae':
    train_dataset_base = utils.TrainDataset_popular(args.dataset)
    train_loader_base = torch.utils.data.DataLoader(train_dataset_base, **params)
else:
    train_dataset_base, train_loader_base = None, None

val_loader_few_shot = torch.utils.data.DataLoader(val_dataset_few_shot, **eval_params)
test_loader_few_shot = torch.utils.data.DataLoader(test_dataset_few_shot, **eval_params)

val_loader_popular = torch.utils.data.DataLoader(val_dataset_popular, **eval_params)
test_loader_popular = torch.utils.data.DataLoader(test_dataset_popular, **eval_params)

###############################################################################
# Build the model
###############################################################################
device = torch.device("cuda" if args.cuda else "cpu")
n_items = train_dataset.n_items
n_users = train_dataset.n_users
best_n100 = - np.inf
modelUserItem = ProtoNetItemUserLL(embed_dim_item=args.embed_dim_item,
                                   embed_dim_user=args.embed_dim_user,
                                   n_users=train_dataset.n_users,
                                   n_memory=args.n_memory,
                                   base_model=args.base_model,
                                   mode='base',
                                   loss_type=args.loss_type,
                                   device=device,
                                   n_items=train_dataset.n_items).to(device)

optimizerUserItem = optim.Adam(modelUserItem.parameters(), lr=args.lr, weight_decay=args.wd)

print("# train batches :", len(train_loader))
args.log_interval = len(train_loader.dataset) // args.classes_per_it

# At any point you can hit Ctrl + C to break out of training early.
def train(model, optimizer, ui_sp_mat=None):
    best_r50 = -np.inf
    try:
        for epoch in range(0, args.epochs):
            epoch_start_time = time.time()

            # Train start.
            model.train()
            train_loss = 0.0
            start_time = time.time()
            batch_idx = 0

            for data in train_loader:
                batch_idx += 1
                loss = 0
                data = [x.to(device, non_blocking=True) for x in data]
                support_items, query_items, idx, mask = data
                model.zero_grad()
                optimizer.zero_grad()

                if model.mode != "base":
                    positive, negative = utils.get_pairs_base(model.item_lookup(idx).cpu())
                else:
                    positive, negative = torch.zeros(1), torch.zeros(1)
                logits, correction= model(support_items, query_items, idx, ui_sp_mat=ui_sp_mat,
                                                    pos_neg=(positive.to(device), negative.to(device)),
                                                    mask=mask)  # B x K x B

                if model.mode == 'base':
                    # targets for VAE-CF.
                    if model.base_model == 'vae' or model.base_model == "dae" or model.base_model == 'bpr':
                        targets = utils.get_target(train_df, torch.arange(n_items).numpy(),
                                                   torch.cat((support_items, query_items),
                                                             dim=1).cpu().numpy()).to(device,
                                                                                      non_blocking=True)
                    else:
                        targets = utils.get_target(train_df, idx.cpu().numpy(),
                                                   torch.cat((support_items, query_items),
                                                             dim=1).cpu().numpy()).to(device,
                                                                                      non_blocking=True)

                else:
                    targets = utils.get_target(train_df, idx.cpu().numpy(), query_items.cpu().numpy()).to(device,
                                                                                                          non_blocking=True)
                pos_weight = utils.get_pos_weight(targets)

                loss += model.loss_function(logits, targets, pos_weight.to(device, non_blocking=True))
                if model.mode == 'few-shot':
                    loss += args.l2_lambda * correction
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                if batch_idx % args.log_interval == 0 and batch_idx > 0:
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                          'loss {:4.2f}'.format(
                        epoch, batch_idx, len(range(0, len(train_loader.dataset), train_loader.batch_size)),
                        elapsed * 1000 / args.log_interval, train_loss / args.log_interval))

                    # Log loss to tensorboard
                    n_iter = epoch * len(range(0, len(train_loader.dataset), args.classes_per_it)) + batch_idx
                    start_time = time.time()
                    train_loss = 0.0

            if epoch % 5 == 0:
                # Evaluate on few-shot validation set, R@20, R@50, N@100.
                print('-' * 89)
                # Evaluate on popular validation set, R@20, R@50, N@100.
                val_loss, n100, r20, r50, query_head, pred_head, pred_head1 = evaluate.evaluate_popular(model, val_loader_popular, n_users,
                                                                        device, ui_sp_mat)

                print('| Popular evaluation | val loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | '
                      'r50 {:4.4f}'.format(val_loss, n100, r20, r50))

                if model.mode == 'few-shot':
                    val_loss, n100, r20, r50, query_fs, pred_fs, pred_fs1 = evaluate.evaluate_few_shot(model, val_loader_few_shot, device,
                                                                             val_query_users, ui_sp_mat)

                    print('| Few-shot evaluation | val loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | '
                          'r50 {:4.4f}'.format(val_loss, n100, r20, r50))

                    print('-' * 89)
                # Save the model if the n100 is the best we've seen so far.
                # global best_n100
                if r50 > best_r50:
                    with open(save, 'wb') as f:
                        torch.save(model, f)
                    best_r50 = r50

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


train_ui_sp_mat = train_dataset_base.train_data_ui if args.base_model == 'vae' or args.base_model == 'dae' else None


if not path.exists("/data/item_weight/{}_{}".format(args.dataset, base_method)) or args.base_training:

    if args.base_model == 'bpr':
        with open(save, 'rb') as f:
            modelUserItem = torch.load(f)
            modelUserItem = modelUserItem.to(device)
        item_embedding = modelUserItem.item_lookup[0].weight
        user_embedding = modelUserItem.user_lookup[0].weight
    elif args.base_model == 'vae':
        train(modelUserItem, optimizerUserItem, train_ui_sp_mat)
        with open(save, 'rb') as f:
            modelUserItem = torch.load(f)
            modelUserItem = modelUserItem.to(device)
        item_embedding = modelUserItem.item_lookup[0].weight
        user_embedding, _, _ = modelUserItem.user_encoder(
            torch.Tensor(train_ui_sp_mat[torch.arange(n_users).numpy(), :].toarray()).to(
                device).squeeze(), torch.arange(n_users).to(device))

        torch.save(modelUserItem.user_encoder, '/data/model/vae_{}_new_full123.pt'.format(args.dataset))
    elif args.base_model == 'dae':
        train(modelUserItem, optimizerUserItem, train_ui_sp_mat)
        with open(save, 'rb') as f:
            modelUserItem = torch.load(f)
            modelUserItem = modelUserItem.to(device)
        item_embedding = modelUserItem.item_lookup[0].weight
        user_embedding, _, _ = modelUserItem.user_encoder(
            torch.Tensor(train_ui_sp_mat[torch.arange(n_users).numpy(), :].toarray()).to(
                device).squeeze(), torch.arange(n_users).to(device))

        torch.save(modelUserItem.user_encoder, '/data/model/dae_{}.pt'.format(args.dataset))

    test_loss, n100, r20, r50, _, _, _ = evaluate.evaluate_popular(modelUserItem, test_loader_popular, n_users,
                                                                      device, train_ui_sp_mat)

    print('=' * 89)
    print('| Popular test evaluation | test loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | '
          'r50 {:4.4f}'.format(test_loss, n100, r20, r50))

    torch.save(item_embedding, "/data/item_weight/{}_{}_new_full".format(args.dataset, base_method))
    torch.save(user_embedding, "/data/user_weight/{}_{}_new".format(args.dataset, base_method))

item_weight = torch.load("/data/item_weight/{}_{}_new_full".format(args.dataset, base_method))
user_weight = torch.load("/data/user_weight/{}_{}_new".format(args.dataset, base_method))

if args.base_model == 'vae':
    user_encoder = torch.load('/data/model/vae_{}_new_full.pt'.format(args.dataset))
elif args.base_model == 'dae':
    user_encoder = torch.load('/data/model/dae_{}.pt'.format(args.dataset))
else:
    user_encoder = None

model_fewshot = ProtoNetItemUserLL(embed_dim_item=args.embed_dim_item,
                                   embed_dim_user=args.embed_dim_user,
                                   n_users=train_dataset.n_users,
                                   n_memory=args.n_memory,
                                   user_embedding=user_weight,
                                   item_embedding=item_weight,
                                   loss_type=args.loss_type,
                                   user_encoder=user_encoder,
                                   model_variant=args.model_variant,
                                   device=device,
                                   mode='few-shot',
                                   base_model=args.base_model,
                                   n_items=train_dataset.n_items).to(device)

optimizer_fewshot = optim.Adam(model_fewshot.parameters(), lr=args.lr, weight_decay=args.wd)
train(model_fewshot, optimizer_fewshot, train_ui_sp_mat)
# Load the best saved model.
with open(save, 'rb') as f:
    model = torch.load(f, map_location=device)
    model = model.to(device)
user_embedding = model.user_lookup(torch.arange(n_users).unsqueeze(1).to(device)).squeeze()

base_item_embedding = model.item_lookup(torch.arange(n_items).unsqueeze(1).to(device)).squeeze().detach()
# Run on test data.

# # Best validation evaluation on few-shot.

val_loss, n100, r20, r50, val_query_fs, val_pred_fs, val_pred_fs1 = evaluate.evaluate_few_shot(model, val_loader_few_shot, device,
                                                                              val_query_users, train_ui_sp_mat, test=True)
print('=' * 89)
print('| Few-shot val evaluation | val loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | '
      'r50 {:4.4f}'.format(val_loss, n100, r20, r50))
# # Best validation evaluation on popular.
val_loss, n100, r20, r50, val_query_head, val_pred_head, val_pred_head1 = evaluate.evaluate_popular(model, val_loader_popular, n_users, device,
                                                                train_ui_sp_mat)
print('=' * 89)
print('| Popular val evaluation | val loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | '
      'r50 {:4.4f}'.format(val_loss, n100, r20, r50))
print('-' * 89)

# Test evaluation on few-shot.

test_loss, n100, r20, r50, test_query_fs, test_pred_fs, test_pred_fs1 = evaluate.evaluate_few_shot(model, test_loader_few_shot,
                                                                                     device,
                                                                                     test_query_users, train_ui_sp_mat, test=True)


print('=' * 89)
print('| Few-shot test evaluation | test loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | '
      'r50 {:4.4f}'.format(test_loss, n100, r20, r50))

# Test evaluation on popular.
test_loss, n100, r20, r50, test_query_head, test_pred_head, test_pred_head1 = evaluate.evaluate_popular(model, test_loader_popular, n_users, device,
                                                                  train_ui_sp_mat)
print('=' * 89)
print('| Popular test evaluation | test loss {:4.4f} | n100 {:4.4f} | r20 {:4.4f} | '
      'r50 {:4.4f}'.format(test_loss, n100, r20, r50))

for lambd in [float(j) / 100 for j in range(5, 100, 5)]:
    joint_n20, joint_n50, joint_r20, joint_r50, all_pred = evaluate.evaluate_joint(model, test_query_fs.to(device),
                                                               test_query_head.to(device),
                                                               test_pred_fs.to(device),
                                                               test_pred_head.to(device),
                                                               test_pred_head1.to(device),
                                                               test_pred_fs1.to(device),
                                                               lambd=lambd)


    print('| Joint test evaluation | n20 {:4.4f} | n50 {:4.4f} | r20 {:4.4f} | '
          'r50 {:4.4f}'.format(joint_n20, joint_n50, joint_r20, joint_r50))

