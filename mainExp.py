

import typing
setattr(__builtins__, 'Union', typing.Union)
setattr(__builtins__, 'Tuple', typing.Tuple)

import sys
sys.path.append("./uni2ts/src")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


import torch
import os
import csv
import numpy as np
from script.datapre import *
from module.model.model import return_model
# from module.method.sv_v2 import ShapleyValue
# from module.method.loo_v2 import LeaveOneOutWithAttack

# from module.method.attack_v5 import GradientReplayAttack, ARIMAAttack, RandomAttack, MoiraiAttack
from module.method.attack_v8 import (
    GradientReplayAttack, ARIMAAttack, RandomAttack, MoiraiAttack,
    OfflineARIMAAttack, AdaptiveARIMAAttack, OfflineMoiraiAttack, OnlineMoiraiAttack,
    frobenius_scaled_identity_provider,   # 便宜版曲率
    # hutchinson_hvp_provider,            # 要真HVP再解开 & 提供 loss_fn/batch
    # zeros_hessian_provider,             # 显式禁用时可用
)


import copy
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--attacker_id', type=int, default=1)
parser.add_argument('--alpha', type=float, default=10)
parser.add_argument('--dataset', type=str, default="tictactoe")
parser.add_argument(
    '--attack_method',
    type=str,
    default="moirai",
    choices=["random", "fedavg", "arima", "arima_adaptive", "moirai", "moirai_online"]
)
parser.add_argument('--client_num', type=int, default=4)
parser.add_argument('--contribution_method',type=str,default='shapley',choices=['shapley', 'loo', 'eds', "cosine"])
parser.add_argument(
    '--use_last_round_sv',
    action=argparse.BooleanOptionalAction,
    default=True,
    help='Use last-round server-gradient Shapley utility (default: True).'
)
parser.add_argument('--use_attack', action='store_true', help='Enable attack or not')
parser.add_argument('--trial_id', type=int, default=0, help='Repetition ID')

# hessian optional
parser.add_argument('--use_hessian', action='store_true', help='Enable Hessian fusion')
parser.add_argument('--lambda_h', type=float, default=0.9, help='Weight for f_P in Eq.(3)')

# adaptive ARIMA options
parser.add_argument('--arima_err_threshold', type=float, default=5e-3,
                    help='Adaptive ARIMA reset threshold on EMA prediction error.')
parser.add_argument('--arima_min_rounds_after_start', type=int, default=2,
                    help='Minimum observed rounds after ARIMA starts before reset is allowed.')
parser.add_argument('--arima_retrain_interval', type=int, default=10,
                    help='Periodic retrain interval for adaptive ARIMA; <=0 disables periodic retraining.')
parser.add_argument('--arima_err_ema_beta', type=float, default=0.9,
                    help='EMA smoothing factor for adaptive ARIMA prediction error.')
parser.add_argument('--arima_trim_history', action=argparse.BooleanOptionalAction, default=True,
                    help='Trim gradient history after adaptive ARIMA reset.')


# lazy mode
parser.add_argument('--use_lazy', action='store_true', help='Enable Q3 lazy mode: sparse honest rounds, fake otherwise')
parser.add_argument('--lazy_honest_gap', type=int, default=3, help='Honest cadence in lazy mode, e.g. 3 means 1 honest / 2 fake')
parser.add_argument('--lazy_honest_phase', type=int, default=0, help='Honest phase offset; default 0 makes round 0 honest')
parser.add_argument('--lazy_fake_policy', type=str, default='attack', choices=['attack', 'zero'], help='Submission policy on non-honest rounds in lazy mode')


# dump clean global-grad history for Moirai fine-tuning 
parser.add_argument('--dump_grad_history', action='store_true',
                    help='Dump flattened global gradients each round (clean run, no attacker)')
parser.add_argument('--dump_dir', type=str, default='data/grad_hist',
                    help='Directory to save gradient CSVs, e.g., data/grad_hist')
parser.add_argument('--results_root', type=str, default='Results',
                    help='Root directory for csv outputs, e.g., Results or Results/Dota2_Random_Fedavg')


args = parser.parse_args()
results_root = args.results_root


# datasets=("tictactoe" "adult" "dota2")
# attack_methods=("random" "fedavg" "arima" "moirai")
# client_nums=(2 4 6 8)
# declare -A alpha_map
# alpha_map[tictactoe]="10 100 1000 20000"
# alpha_map[adult]="1000 100000 1000000 20000000"
# alpha_map[dota2]="10000 1000000 50000000 200000000"

attacker_id = args.attacker_id
alpha = args.alpha
dataset = args.dataset.lower()
attack_method = args.attack_method.lower()
num_parts = args.client_num
use_attack = args.use_attack
trial_id = args.trial_id

need_history = (args.contribution_method == "eds")
# need_gradients = args.contribution_method in ["shapley", "loo", "cosine"]
need_gradients = (
    args.contribution_method in ["shapley", "loo", "cosine"]
    or attack_method in ["fedavg", "arima", "arima_adaptive", "moirai", "moirai_online"]
)

# datasets=("tictactoe" "adult" "dota2")
# attack_methods=("random" "fedavg" "arima" "moirai")
# client_nums=(2 4 6 8)
# declare -A alpha_map
# alpha_map[tictactoe]="10 100 1000 20000"
# alpha_map[adult]="1000 100000 1000000 20000000"
# alpha_map[dota2]="10000 1000000 50000000 200000000"
# alpha_map[mnist] = [10, 1000, 100000, 10000000] 

uniform_thresholds = {
    "tictactoe": 20000,
    "adult": 20000000,
    "dota2": 200000000,
    "mnist": 10000000,
    "imdb": 10000000,  # 修改！ 确认alpha！ imdb alpha还未确定***
}

distribution = "uniform" if alpha == uniform_thresholds[dataset] else "quantity skew"

# model_name_map = {
#     "tictactoe": "TicTacToeLR",
#     "adult": "AdultLR",
#     "dota2": "Dota2LR",
#     "mnist": "MNISTLR"
# }
model_name_map = {
    "tictactoe": "TicTacToeMLP",
    "adult": "AdultMLP",
    "dota2": "Dota2MLP",
    "mnist": "MNISTMLP",
    "imdb": "IMDBMLP",
}
model_name = model_name_map[dataset]



# ======== Configuration ========
seed = 42
num_epoch = 50
num_local_epochs = 1

lr = 0.008
hidden_layer_size = 16
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

attack_time_log = []



# Fixed window for SV evaluation (start from checkpoint at T-R_FOR_SV)
R_FOR_SV = 20
# ======== Nohup Logs ========
print(f"=== Running {dataset}, attack={attack_method}, N={num_parts}, alpha={alpha}, "
      f"use_attack={use_attack}, attacker_id={attacker_id}, trial_id={trial_id} ===", flush=True)


if device.type == 'cuda':
    print(f"detect available GPU: {torch.cuda.get_device_name(0)}, FL training is running in pure GPU mode.")
else:
    print("Warning: No available GPU detected. The training process will revert to CPU mode.")

# ======== Attack Configuration ========
K = 5              # look back rounds
WARMUP = 10        # warmup rounds / random rounds
HISTORY = K        # Moirai's history_length = k 



# Load Data & Model
loader = get_data(seed=seed, dataset=dataset, distribution=distribution,
                  alpha=alpha, num_parts=num_parts)

model_kwargs = dict(
    seed=seed,
    num_epoch=num_epoch,
    lr=lr,
    batch_size=batch_size,
    hidden_layer_size=hidden_layer_size,
    device=device,
    dataset=dataset,
)
if dataset == "imdb":
    model_kwargs["input_size"] = int(loader.X_train.shape[1])
model = return_model(model_name, **model_kwargs)

model.init_args = {
    "seed": seed,
    "num_epoch": num_epoch,
    "lr": lr,
    "device": device,
    "hidden_layer_size": hidden_layer_size,
    "batch_size": batch_size
}
if dataset == "imdb":
    model.init_args["input_size"] = int(loader.X_train.shape[1])



# VLDB-aligned: save common initialization for SV
init_state = copy.deepcopy(model.state_dict())
def prepare_loader_without_attacker(loader, attacker_id):
    ids = [i for i in range(len(loader.X_train_parts)) if i != attacker_id]
    new_loader = copy.deepcopy(loader)
    new_loader.X_train_parts = [loader.X_train_parts[i] for i in ids]
    new_loader.y_train_parts = [loader.y_train_parts[i] for i in ids]
    return new_loader, ids


def _assert_all_grads_on_device(grads_by_client, device, tag=""):
    # grads_by_client: List[List[Tensor]]
    for ci, grad_list in enumerate(grads_by_client):
        for pi, g in enumerate(grad_list):
            if not isinstance(g, torch.Tensor):
                raise RuntimeError(f"[{tag}] grad is not a Tensor: client={ci} param={pi} type={type(g)}")
            if g.device != device:
                raise RuntimeError(
                    f"[{tag}] grad device mismatch: client={ci} param={pi} g.device={g.device} expected={device}"
                )

def _assert_model_params_on_device(model, device, tag=""):
    for pi, p in enumerate(model.parameters()):
        if p.device != device:
            raise RuntimeError(
                f"[{tag}] param device mismatch: param={pi} p.device={p.device} expected={device}"
            )

def _assert_weights_on_device(weights, device, tag=""):
    # weights can be Tensor on GPU or Python floats; both ok
    if isinstance(weights, torch.Tensor) and weights.device != device:
        raise RuntimeError(f"[{tag}] weights.device={weights.device} expected={device}")


def _minmax_normalize_rows(contribs, eps=1e-12):
    if contribs is None or len(contribs) == 0:
        return contribs
    arr = np.asarray(contribs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row_min = arr.min(axis=1, keepdims=True)
    row_max = arr.max(axis=1, keepdims=True)
    denom = np.maximum(row_max - row_min, eps)
    return ((arr - row_min) / denom).tolist()



def run_federated_learning(loader, model, attacker_id, attacker, use_attack,
                         num_epoch, num_local_epochs, device, record_gradients=False, collect_history=False):
    # Snapshot used by windowed-SV: T-R_FOR_SV checkpoint
    state_t_minus_R = None  # model state right before the last R rounds
    num_parts = len(loader.X_train_parts)
    client_models = [copy.deepcopy(model) for _ in range(num_parts)]
    shard_sizes = torch.tensor([len(loader.X_train_parts[i]) for i in range(num_parts)], dtype=torch.float, device=device)
    weights = (shard_sizes / shard_sizes.sum())
    attack_time_log = []
    # [PATCH] Last-round SV caches
    state_t_minus_1 = None
    last_round_client_grads = None
    last_round_weights = None
    global_accuracies = []
    
    # only initialize when EDS is needed 
    round_client_gradients = []  # for EDS: List[List[torch.Tensor]]
    round_global_gradients = []  # for EDS: List[List[torch.Tensor]]
    attack_global_gradient_history = []  # for ARIMA/Moirai: List[np.ndarray]

    for round_num in range(num_epoch):
            # Take snapshot right before entering the last R_FOR_SV rounds
        trig = (num_epoch - R_FOR_SV) - 1
        if num_epoch >= R_FOR_SV and round_num == trig:
            state_t_minus_R = copy.deepcopy(model.state_dict())
        current_round_client_grads = []
        

        for i in range(num_parts):
            X_i, y_i = loader.X_train_parts[i], loader.y_train_parts[i]
            model_i = client_models[i].to(device)
            backup = copy.deepcopy(model_i)

            # local train
            model_i.fit(X_i, y_i, incremental=True, num_epochs=num_local_epochs)
            true_gradient = [new.data - old.data for new, old in zip(model_i.parameters(), backup.parameters())]
            
            if use_attack and i == attacker_id:
                # Q2 (use_lazy=False): attacker always fakes.
                # Q3 (use_lazy=True): honest only on sparse rounds (1 honest / N fake).
                is_honest_round = args.use_lazy and (
                    ((round_num - args.lazy_honest_phase) % max(1, args.lazy_honest_gap)) == 0
                )

                # log
                print(
                    f"[ROUND {round_num}] client={i} honest={bool(is_honest_round)} "
                    f"method={args.attack_method} "
                    f"hessian={'on' if args.use_hessian else 'off'} "
                    f"lambda_h={args.lambda_h if args.use_hessian else 'n/a'} "
                    f"lazy={'on' if args.use_lazy else 'off'} "
                    f"(honest_gap={args.lazy_honest_gap if args.use_lazy else 'n/a'}, "
                    f"honest_phase={args.lazy_honest_phase if args.use_lazy else 'n/a'}, "
                    f"fake_policy={args.lazy_fake_policy if args.use_lazy else 'n/a'})",
                    flush=True
                )

                if is_honest_round:
                    gradient = true_gradient
                    print(f"[LAZY] round={round_num} client={i} submit honest", flush=True)
                else:
                    if args.use_lazy and args.lazy_fake_policy == 'zero':
                        gradient = [torch.zeros_like(p, device=p.device) for p in model.parameters()]
                        print(f"[LAZY] round={round_num} client={i} submit zero", flush=True)
                    else:
                        t0 = time.time()
                        gradient = attacker.get_fake_gradient(round_num, device, model_i)
                        attack_time_log.append(time.time() - t0)

                        # report where the fake gradient came from
                        src = "predict" if args.attack_method in ["arima", "arima_adaptive", "moirai", "moirai_online"] else "avg"
                        hist_len = len(getattr(attacker, "global_gradient_history", []))
                        print(f"[ATTACK] round={round_num} client={i} source={src} history_len={hist_len}", flush=True)
            else:
                # non-attacking period / no attacking：submit real gradient
                gradient = true_gradient

            # after choosing `gradient` (true or fake)
            _assert_all_grads_on_device([gradient], device, tag=f"round={round_num}/client={i}/submit")
            current_round_client_grads.append(gradient)




        _assert_all_grads_on_device(current_round_client_grads, device, tag=f"round={round_num}/before_agg")
        _assert_model_params_on_device(model, device, tag=f"round={round_num}/before_agg")
        _assert_weights_on_device(weights, device, tag=f"round={round_num}/before_agg")

        # FedAvg Aggregation
        aggregated_gradient = [torch.zeros_like(param).to(device) for param in model.parameters()]
        for grad, weight in zip(current_round_client_grads, weights):
            for ag, g in zip(aggregated_gradient, grad):
                ag += g * weight

        # Always flatten and (optionally) dump the global aggregated gradient as one CSV row
        _assert_all_grads_on_device([aggregated_gradient], device, tag=f"round={round_num}/after_agg")
        _assert_model_params_on_device(model, device, tag=f"round={round_num}/after_agg")

        # flat_global_grad = torch.cat([g.view(-1) for g in aggregated_gradient]).detach().numpy()
        flat_global_grad = torch.cat([g.reshape(-1) for g in aggregated_gradient]).detach()
        flat_global_grad = None
        if args.dump_grad_history:
            _assert_all_grads_on_device([aggregated_gradient], device, tag=f"round={round_num}/after_agg")
            flat_global_grad = torch.cat([g.view(-1) for g in aggregated_gradient]).detach().cpu().numpy()



        if use_attack:
            attacker.record_global_gradient(aggregated_gradient)

        if args.dump_grad_history:
            # data/grad_hist/<dataset>/run_<trial_id>.csv
            os.makedirs(os.path.join(args.dump_dir, dataset), exist_ok=True)
            csv_path = os.path.join(args.dump_dir, dataset, f'run_{trial_id}.csv')

            if not os.path.exists(csv_path):
                with open(csv_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    header = ['t'] + [f'd{i}' for i in range(flat_global_grad.shape[0])]
                    w.writerow(header)

            with open(csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([round_num] + flat_global_grad.tolist())


        # update: aggregated gradient --> global model 
        # Cache last-round state and gradients before applying update
        if round_num == num_epoch - 1:
            # 1) snapshot of global model before applying round T update
            state_t_minus_1 = copy.deepcopy(model.state_dict())
            # 2) gradients submitted by each client in round T
            last_round_client_grads = {
                i: [g.detach().clone().to(device) for g in grad_list]
                for i, grad_list in enumerate(current_round_client_grads)
            }
            # 3) aggregation weights of round T (by shard size)
            last_round_weights = weights.tolist()

        _assert_model_params_on_device(model, device, tag=f"round={round_num}/before_apply")
        for param, update in zip(model.parameters(), aggregated_gradient):
            param.data += update.data
        for cm in client_models:
            cm.load_state_dict(model.state_dict())

        # evaluation
        acc = model.score(loader.X_test, loader.y_test, ["accuracy"])[0]
        global_accuracies.append(acc)
        print(f"Round {round_num}: Global Accuracy = {acc:.4f}")

        if collect_history:
            # Stores the client gradient for the current round
            round_client_gradients.append(current_round_client_grads)
            # Stores the global gradient of the current round
            # round_global_gradients.append([g.detach().clone().cpu() for g in aggregated_gradient])
            round_global_gradients.append([g.detach().clone() for g in aggregated_gradient])

    return_values = (model, attack_time_log, global_accuracies)
    if collect_history:
        return_values += (round_client_gradients, round_global_gradients)
    return_values += (state_t_minus_1, last_round_client_grads, last_round_weights)
    
    return_values += (state_t_minus_R,)
    return return_values

# Load ContributionCalculator

if args.contribution_method == "shapley":
        from module.method.sv_v7 import ShapleyValue as ContributionCalculator 
elif args.contribution_method == "loo":
    from module.method.loo_v4 import LeaveOneOutWithAttack as ContributionCalculator
elif args.contribution_method == "eds":
    from module.method.eds_v3 import EDSCalculator
# elif args.contribution_method in ["cosine"]:
#     from module.method.simple_metrics import SimpleMetricsCalculator as ContributionCalculator
else:
    raise ValueError("Invalid contribution method")


# hessian
provider = frobenius_scaled_identity_provider if args.use_hessian else None
lam = args.lambda_h if args.use_hessian else 1.0


#  6 attacking algorithms
if use_attack:
    if attack_method == "random":
        attacker = RandomAttack()
    elif attack_method == "fedavg":
        attacker = GradientReplayAttack(k=K, start_attack_round=WARMUP)
    elif attack_method == "arima":
        attacker = OfflineARIMAAttack(
            k=K, random_round=WARMUP, model_params=list(model.parameters()),
            lambda_h=lam, hessian_provider=provider
        )
    elif attack_method == "arima_adaptive":
        attacker = AdaptiveARIMAAttack(
            k=K, random_round=WARMUP, model_params=list(model.parameters()),
            lambda_h=lam, hessian_provider=provider,
            err_ema_beta=args.arima_err_ema_beta,
            err_threshold=args.arima_err_threshold,
            min_rounds_after_start=args.arima_min_rounds_after_start,
            retrain_interval=args.arima_retrain_interval,
            trim_history=args.arima_trim_history,
        )
    elif attack_method == "moirai":
        attacker = OfflineMoiraiAttack(
            k=K, random_round=WARMUP, history_length=HISTORY,
            lambda_h=lam, hessian_provider=provider
        )
    elif attack_method == "moirai_online":
        attacker = OnlineMoiraiAttack(
            k=K, random_round=WARMUP, history_length=HISTORY,
            lambda_h=lam, hessian_provider=provider
        )

    # elif attack_method == "arima":
    #     # attacker = ARIMAAttack(k=4, random_round=10)
    #     attacker = ARIMAAttack(k=4, random_round=10, model_params=list(model.parameters()))
    # elif attack_method == "moirai":
    #     attacker = MoiraiAttack(k=4, random_round=10)



    # Run FL with attacker
    model_with_attacker, attack_time_log_with_attacker, global_accuracies_log_with_attacker, *history_with = run_federated_learning(
        loader, model, attacker_id, attacker,
        use_attack=True,
        num_epoch=num_epoch,
        num_local_epochs=num_local_epochs,
        device=device,
        record_gradients=need_gradients,
        collect_history=need_history
    )
    # unpack last-round caches from history_with
    if need_history:
        round_client_grads, round_global_grads, state_t_minus_1, last_round_client_grads, last_round_weights, state_t_minus_R = history_with
    else:
        state_t_minus_1, last_round_client_grads, last_round_weights, state_t_minus_R = history_with

    # Run FL no attacker
    subset_loader, subset_ids = prepare_loader_without_attacker(loader, attacker_id)
    subset_model_kwargs = dict(
        seed=seed,
        num_epoch=num_epoch,
        lr=lr,
        batch_size=batch_size,
        hidden_layer_size=hidden_layer_size,
        device=device,
        dataset=dataset,
    )
    if dataset == "imdb":
        subset_model_kwargs["input_size"] = int(subset_loader.X_train.shape[1])
    model_no_attacker = return_model(model_name, **subset_model_kwargs)
    model_no_attacker.init_args = model.init_args  # keep hyperparams consistent
    model_without_attacker, attack_time_log_without_attacker, global_accuracies_log_without_attacker, *history_without = run_federated_learning(
        subset_loader, model_no_attacker, attacker_id=None, attacker=None,
        use_attack=False, num_epoch=num_epoch, num_local_epochs=num_local_epochs,
        device=device, record_gradients=need_gradients, collect_history=need_history
    )
    if need_history:
        round_client_grads_wo, round_global_grads_wo, state_t_minus_1_wo, last_round_client_grads_wo, last_round_weights_wo, state_t_minus_R_wo = history_without
    else:
        state_t_minus_1_wo, last_round_client_grads_wo, last_round_weights_wo, state_t_minus_R_wo = history_without


    # Contribution estimation
    contribs_w_minmax = [[]]
    contribs_wo_minmax = [[]]
    eds_result = None
    if args.contribution_method == "eds":
        # Paper-style EDS detection:
        # pairwise client distances each round + KMeans + multi-round persistence
        eds_result = EDSCalculator.detect_free_riders_from_history(
            round_client_gradients=round_client_grads,
            dynamic_weighting=False,
            warmup_rounds=0,
            persistence_ratio=0.5,
            min_suspicious_rounds=2,
        )
        suspicious_clients = eds_result["suspicious_clients"]
        suspicious_counts = eds_result["suspicious_counts"]
        round_records = eds_result["round_records"]
        threshold = eds_result["decision_threshold"]

        if round_records:
            last_record = round_records[-1]
            print(
                f"[EDS Round {last_record['round_idx']}] "
                f"scores={last_record['client_scores']} "
                f"suspicious={last_record['suspicious_clients']} "
                f"weights={last_record['weights']}"
            )
        print(
            f"[EDS Detection] suspicious_counts={suspicious_counts} "
            f"threshold={threshold} final_suspicious={suspicious_clients} "
            f"attacker? {attacker_id in suspicious_clients}"
        )
        contribs_w = [[float("nan")] * num_parts]
        contribs_wo = [[float("nan")] * (num_parts - 1)]
        contribs_w_minmax = [[float("nan")] * num_parts]
        contribs_wo_minmax = [[float("nan")] * (num_parts - 1)]
    else:
        # Shapley / LOO / Cosine: compute with- and without-attacker
        if args.contribution_method == "shapley":
            shapley_kwargs_with = dict(
                attacker=attacker,
                attacker_id=attacker_id,
                use_last_round_sv=args.use_last_round_sv,
                last_round_state=state_t_minus_1,
                last_round_client_grads=last_round_client_grads,
                last_round_weights=last_round_weights,
                mini_local_epochs=num_local_epochs,
                use_lazy=args.use_lazy,
                lazy_honest_gap=args.lazy_honest_gap,
                lazy_honest_phase=args.lazy_honest_phase,
                lazy_fake_policy=args.lazy_fake_policy,
            )
            if not args.use_last_round_sv:
                shapley_kwargs_with.update(
                    base_state_dict=init_state,
                    mini_rounds=num_epoch,
                )
            contrib_with = ContributionCalculator(
                loader, model_with_attacker, {},
                ["accuracy"],
                **shapley_kwargs_with
            )
        else:
            contrib_with = ContributionCalculator(
                loader, model_with_attacker, {},
                ["accuracy"],
                attacker=attacker,
                attacker_id=attacker_id,
                base_state_dict=init_state,
                mini_rounds=num_epoch,
                mini_local_epochs=num_local_epochs,
                use_lazy=args.use_lazy,
                lazy_honest_gap=args.lazy_honest_gap,
                lazy_honest_phase=args.lazy_honest_phase,
                lazy_fake_policy=args.lazy_fake_policy,
            )
        contribs_w, _ = contrib_with.get_contributions()

        if args.contribution_method == "shapley":
            shapley_kwargs_without = dict(
                attacker=None,
                attacker_id=None,
                use_last_round_sv=args.use_last_round_sv,
                last_round_state=state_t_minus_1_wo,
                last_round_client_grads=last_round_client_grads_wo,
                last_round_weights=last_round_weights_wo,
                mini_local_epochs=num_local_epochs,
                use_lazy=False,
                lazy_honest_gap=args.lazy_honest_gap,
                lazy_honest_phase=args.lazy_honest_phase,
                lazy_fake_policy=args.lazy_fake_policy,
            )
            if not args.use_last_round_sv:
                shapley_kwargs_without.update(
                    base_state_dict=init_state,
                    mini_rounds=num_epoch,
                )
            contrib_without = ContributionCalculator(
                subset_loader, model_no_attacker, {},
                ["accuracy"],
                **shapley_kwargs_without
            )
        else:
            contrib_without = ContributionCalculator(
                subset_loader, model_no_attacker, {},
                ["accuracy"],
                attacker=None,
                attacker_id=None,
                base_state_dict=init_state,
                mini_rounds=num_epoch,
                mini_local_epochs=num_local_epochs,
                use_lazy=False,
                lazy_honest_gap=args.lazy_honest_gap,
                lazy_honest_phase=args.lazy_honest_phase,
                lazy_fake_policy=args.lazy_fake_policy,
            )
        contribs_wo, _ = contrib_without.get_contributions()
        if args.contribution_method in ["shapley", "loo"]:
            contribs_w_minmax = _minmax_normalize_rows(contribs_w)
            contribs_wo_minmax = _minmax_normalize_rows(contribs_wo)
        else:
            contribs_w_minmax = contribs_w
            contribs_wo_minmax = contribs_wo


else:
    # === No attacker path ===
    # model_no_attack, attack_time_log_no_attack, global_accuracies_log_no_attack = run_federated_learning(
    model_no_attack, attack_time_log_no_attack, global_accuracies_log_no_attack, state_t_minus_1_no, last_round_client_grads_no, last_round_weights_no, state_t_minus_R_no = run_federated_learning(
        loader, model, attacker_id=None, attacker=None, use_attack=False,
        num_epoch=num_epoch, num_local_epochs=num_local_epochs,
        device=device, record_gradients=(args.contribution_method in ["shapley", "loo"]), collect_history=False
    )
    contribs_w_minmax = [[]]
    contribs_wo_minmax = [[]]
    eds_result = None
    if args.contribution_method in ["shapley", "loo", "cosine"]:
        if args.contribution_method == "shapley":
            shapley_kwargs_no_attack = dict(
                use_last_round_sv=args.use_last_round_sv,
                last_round_state=state_t_minus_1_no,
                last_round_client_grads=last_round_client_grads_no,
                last_round_weights=last_round_weights_no,
                mini_local_epochs=num_local_epochs,
                use_lazy=False,
                lazy_honest_gap=args.lazy_honest_gap,
                lazy_honest_phase=args.lazy_honest_phase,
                lazy_fake_policy=args.lazy_fake_policy,
            )
            if not args.use_last_round_sv:
                shapley_kwargs_no_attack.update(
                    base_state_dict=init_state,
                    mini_rounds=num_epoch,
                )
            contrib_without = ContributionCalculator(
                loader, model_no_attack, {},
                ["accuracy"],
                **shapley_kwargs_no_attack
            )
        else:
            contrib_without = ContributionCalculator(
                loader, model_no_attack, {},
                ["accuracy"],
                attacker=None,
                attacker_id=None,
                base_state_dict=init_state,
                mini_rounds=num_epoch,
                mini_local_epochs=num_local_epochs,
                use_lazy=False,
                lazy_honest_gap=args.lazy_honest_gap,
                lazy_honest_phase=args.lazy_honest_phase,
                lazy_fake_policy=args.lazy_fake_policy,
            )
        contribs_wo, _ = contrib_without.get_contributions()
        if args.contribution_method in ["shapley", "loo"]:
            contribs_wo_minmax = _minmax_normalize_rows(contribs_wo)
        else:
            contribs_wo_minmax = contribs_wo
    else:
        contribs_wo = [[]]
        contribs_wo_minmax = [[]]

contribution_dir = os.path.join(results_root, f"Contribution_{args.contribution_method}")
time_dir = os.path.join(results_root, f"Time_{args.contribution_method}")
accuracy_dir = os.path.join(results_root, f"Accuracy_{args.contribution_method}")

os.makedirs(contribution_dir, exist_ok=True)
os.makedirs(time_dir, exist_ok=True)
os.makedirs(accuracy_dir, exist_ok=True)



with open(os.path.join(contribution_dir, f"combined_contributions_trial{trial_id}.csv"), "a", newline="") as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow([
            "Dataset","AttackMethod","Alpha","ClientNum","AttackerID","TrialID",
            "UseAttack","ContributionMethod",
            "UseHessian","UseLazy","LazyHonestGap","LazyHonestPhase","LazyFakePolicy",
            "ClientID",
            "WithAttacker","WithoutAttacker","WithAttackerMinMax","WithoutAttackerMinMax"
        ])

    for i in range(num_parts):
        if use_attack:
            w_val = contribs_w[0][i]
            w_val_minmax = contribs_w_minmax[0][i]
            if i == attacker_id:
                wo_val = "N/A"
                wo_val_minmax = "N/A"
            else:
                j = i if i < attacker_id else i - 1
                wo_val = contribs_wo[0][j]
                wo_val_minmax = contribs_wo_minmax[0][j]
        else:
            w_val = "N/A"
            w_val_minmax = "N/A"
            wo_val = contribs_wo[0][i]
            wo_val_minmax = contribs_wo_minmax[0][i]

        writer.writerow([
            dataset, attack_method, alpha, num_parts, attacker_id, trial_id,
            use_attack, args.contribution_method,
            args.use_hessian, args.use_lazy, args.lazy_honest_gap, args.lazy_honest_phase, args.lazy_fake_policy,
            i,
            w_val, wo_val, w_val_minmax, wo_val_minmax
        ])


with open(os.path.join(time_dir, f"attacking_time_log_trial{trial_id}.csv"), "a", newline="") as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow([
            "Dataset","AttackMethod","Alpha","ClientNum","AttackerID","TrialID",
            "UseAttack","ContributionMethod",
            "UseHessian","UseLazy","LazyHonestGap","LazyHonestPhase","LazyFakePolicy",
            "WithAttackerSec","WithoutAttackerSec"
        ])

    base = [
        dataset, attack_method, alpha, num_parts, attacker_id, trial_id,
        use_attack, args.contribution_method,
        args.use_hessian, args.use_lazy, args.lazy_honest_gap, args.lazy_honest_phase, args.lazy_fake_policy
    ]

    if use_attack:
        writer.writerow(base + [sum(attack_time_log_with_attacker), sum(attack_time_log_without_attacker)])
    else:
        writer.writerow(base + ["N/A", "N/A"])



with open(os.path.join(accuracy_dir, f"global_accuracy_log_trial{trial_id}.csv"), "a", newline="") as f:
    writer = csv.writer(f)

    if f.tell() == 0:
        writer.writerow([
            "Dataset","AttackMethod","Alpha","ClientNum","AttackerID","TrialID",
            "UseAttack","ContributionMethod",
            "UseHessian","UseLazy","LazyHonestGap","LazyHonestPhase","LazyFakePolicy",
            "Round","WithAttackerAccuracy","WithoutAttackerAccuracy"
        ])

    for i in range(num_epoch):
        if use_attack:
            acc_with = global_accuracies_log_with_attacker[i] if i < len(global_accuracies_log_with_attacker) else ""
            acc_without = global_accuracies_log_without_attacker[i] if i < len(global_accuracies_log_without_attacker) else ""
        else:
            acc_with = "N/A"
            acc_without = global_accuracies_log_no_attack[i] if i < len(global_accuracies_log_no_attack) else ""

        writer.writerow([
            dataset, attack_method, alpha, num_parts, attacker_id, trial_id,
            use_attack, args.contribution_method,
            args.use_hessian, args.use_lazy, args.lazy_honest_gap, args.lazy_honest_phase, args.lazy_fake_policy,
            i, acc_with, acc_without
        ])

if args.contribution_method == "eds":
    with open(os.path.join(contribution_dir, f"eds_detection_trial{trial_id}.csv"), "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "Dataset", "AttackMethod", "Alpha", "ClientNum", "AttackerID", "TrialID",
                "UseAttack", "ContributionMethod",
                "UseHessian", "UseLazy", "LazyHonestGap", "LazyHonestPhase", "LazyFakePolicy",
                "DetectedSuspiciousClients", "DetectedAttackerHit",
                "SuspiciousCounts", "DecisionThreshold", "EffectiveRounds",
                "LastRoundIdx", "LastRoundScores", "LastRoundSuspiciousClients", "LastRoundWeights"
            ])

        last_round_idx = ""
        last_round_scores = ""
        last_round_suspicious = ""
        last_round_weights = ""
        suspicious_clients = []
        suspicious_counts = {}
        threshold = ""
        effective_rounds = ""

        if eds_result is not None:
            suspicious_clients = eds_result.get("suspicious_clients", [])
            suspicious_counts = eds_result.get("suspicious_counts", {})
            threshold = eds_result.get("decision_threshold", "")
            effective_rounds = eds_result.get("effective_rounds", "")
            round_records = eds_result.get("round_records", [])
            if round_records:
                last_record = round_records[-1]
                last_round_idx = last_record.get("round_idx", "")
                last_round_scores = last_record.get("client_scores", {})
                last_round_suspicious = last_record.get("suspicious_clients", [])
                last_round_weights = last_record.get("weights", {})

        writer.writerow([
            dataset, attack_method, alpha, num_parts, attacker_id, trial_id,
            use_attack, args.contribution_method,
            args.use_hessian, args.use_lazy, args.lazy_honest_gap, args.lazy_honest_phase, args.lazy_fake_policy,
            str(suspicious_clients), int(attacker_id in suspicious_clients) if use_attack else "N/A",
            str(suspicious_counts), threshold, effective_rounds,
            last_round_idx, str(last_round_scores), str(last_round_suspicious), str(last_round_weights)
        ])
