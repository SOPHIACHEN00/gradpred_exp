import copy
import time

import torch

from module.method.measure import Measure


class LeaveOneOutWithAttack(Measure):
    name = "LeaveOneOut"

    def __init__(
        self,
        loader,
        model,
        cache,
        value_functions,
        attacker=None,
        attacker_id=None,
        base_state_dict=None,
        mini_rounds: int | None = None,
        mini_local_epochs: int | None = None,
        use_lazy: bool = False,
        lazy_honest_gap: int = 3,
        lazy_honest_phase: int = 0,
        lazy_fake_policy: str = "attack",
    ):
        super().__init__(loader, model, cache, value_functions=value_functions)
        self.name = self.__class__.name
        self.cache[str(set())] = [0 for _ in range(len(value_functions))]

        self.attacker = attacker
        self.attacker_id = attacker_id
        self.base_state_dict = copy.deepcopy(base_state_dict) if base_state_dict is not None else None
        self.mini_rounds = int(self.model.num_epoch if mini_rounds is None else mini_rounds)
        self.mini_local_epochs = int(1 if mini_local_epochs is None else mini_local_epochs)

        self.use_lazy = bool(use_lazy)
        self.lazy_honest_gap = int(lazy_honest_gap)
        self.lazy_honest_phase = int(lazy_honest_phase)
        self.lazy_fake_policy = str(lazy_fake_policy)

    def get_contributions(self, **kwargs):
        device = self.model.device
        t0 = time.time()

        all_parts = set(range(self.num_parts))
        for idx_val, value_function in enumerate(self.value_functions):
            baseline_value = self.evaluate_subset_world(all_parts, value_function)
            for i in range(self.num_parts):
                subset = set(all_parts)
                subset.discard(i)
                removed_value = self.evaluate_subset_world(subset, value_function)
                self.contributions[idx_val][i] = baseline_value - removed_value

        if "cuda" in str(device):
            torch.cuda.synchronize()

        t_cal = time.time() - t0
        return self.contributions.tolist(), t_cal

    def evaluate_subset_world(self, parts: set, value_function: str):
        assert isinstance(parts, set)
        assert isinstance(value_function, str)
        if (self.attacker is not None) and (self.attacker_id is not None) and (self.attacker_id in parts):
            return self.evaluate_subset_with_attack(parts, value_function)
        return self.evaluate_subset(parts, value_function)

    def _load_base_or_initial(self):
        if self.base_state_dict is not None:
            self.model.load_state_dict(copy.deepcopy(self.base_state_dict))
        elif hasattr(self.model, "initial_state_dict"):
            self.model.load_state_dict(copy.deepcopy(self.model.initial_state_dict))

    def _make_fresh_attacker(self):
        if self.attacker is None:
            return None

        attacker = copy.deepcopy(self.attacker)

        if hasattr(attacker, "global_gradient_history"):
            attacker.global_gradient_history = []
        if hasattr(attacker, "arima_started"):
            attacker.arima_started = False
        if hasattr(attacker, "arima_start_round"):
            attacker.arima_start_round = None
        if hasattr(attacker, "offline_model_trained"):
            attacker.offline_model_trained = False
        if hasattr(attacker, "offline_model"):
            attacker.offline_model = {}
        if hasattr(attacker, "_err_ema"):
            attacker._err_ema = None
        if hasattr(attacker, "_since_start"):
            attacker._since_start = 0
        if hasattr(attacker, "_last_pred_flat"):
            attacker._last_pred_flat = None
        if hasattr(attacker, "_last_pred_round"):
            attacker._last_pred_round = None

        return attacker

    def _is_lazy_honest_round(self, round_num: int) -> bool:
        if not self.use_lazy:
            return False
        gap = max(1, self.lazy_honest_gap)
        return ((round_num - self.lazy_honest_phase) % gap) == 0

    def _simulate_full_fl(self, parts: set, attacker_override=None):
        device = getattr(self.model, "device", torch.device("cpu"))
        plist = sorted(list(parts))
        if len(plist) == 0:
            return

        shard_sizes = torch.tensor(
            [len(self.X_train_parts[i]) for i in plist],
            dtype=torch.float,
            device=device,
        )
        weights = shard_sizes / shard_sizes.sum()

        client_models = [copy.deepcopy(self.model) for _ in plist]
        for cm in client_models:
            cm.to(device)

        attacker_in = (attacker_override is not None) and (self.attacker_id in plist)
        round_count = int(self.mini_rounds)
        local_epochs = int(self.mini_local_epochs)

        for round_num in range(round_count):
            current_round_updates = []

            for local_k, cid in enumerate(plist):
                X_i, y_i = self.X_train_parts[cid], self.y_train_parts[cid]
                model_i = client_models[local_k].to(device)
                backup = [p.detach().clone() for p in model_i.parameters()]
                model_i.fit(X_i, y_i, incremental=True, num_epochs=local_epochs)
                true_update = [p.detach() - b for p, b in zip(model_i.parameters(), backup)]

                if attacker_in and cid == self.attacker_id:
                    is_honest_round = self._is_lazy_honest_round(round_num)
                    if is_honest_round:
                        use_update = true_update
                    elif self.use_lazy and self.lazy_fake_policy == "zero":
                        use_update = [torch.zeros_like(p, device=device) for p in self.model.parameters()]
                    else:
                        use_update = attacker_override.get_fake_gradient(
                            round_num=round_num,
                            device=device,
                            model=model_i,
                        )
                else:
                    use_update = true_update

                current_round_updates.append(use_update)

            aggregated = [torch.zeros_like(p, device=device) for p in self.model.parameters()]
            for upd, weight in zip(current_round_updates, weights):
                for acc, u in zip(aggregated, upd):
                    acc.add_(u, alpha=float(weight))

            if attacker_in and hasattr(attacker_override, "record_global_gradient"):
                attacker_override.record_global_gradient(
                    [g.detach().clone() for g in aggregated]
                )

            with torch.no_grad():
                for p, u in zip(self.model.parameters(), aggregated):
                    p.add_(u)

            sdict = self.model.state_dict()
            for client_model in client_models:
                client_model.load_state_dict(sdict)

    def evaluate_subset(self, parts: set, value_function: str):
        idx = self.value_functions.index(value_function)
        key = str(set(sorted(parts)))
        if key in self.cache:
            return self.cache[key][idx]

        self._load_base_or_initial()
        self._simulate_full_fl(parts, attacker_override=None)

        vals = self.model.score(self.X_test, self.y_test, self.value_functions)
        if key not in self.cache:
            self.cache[key] = [0 for _ in range(len(self.value_functions))]
        for val_idx, val in enumerate(vals):
            self.cache[key][val_idx] = val
        return self.cache[key][idx]

    def evaluate_subset_with_attack(self, parts: set, value_function: str):
        idx = self.value_functions.index(value_function)
        key = str(set(sorted(parts))) + "#atk"
        if key in self.cache:
            return self.cache[key][idx]

        attacker_clone = self._make_fresh_attacker()
        self._load_base_or_initial()
        self._simulate_full_fl(parts, attacker_override=attacker_clone)

        vals = self.model.score(self.X_test, self.y_test, self.value_functions)
        if key not in self.cache:
            self.cache[key] = [0 for _ in range(len(self.value_functions))]
        for val_idx, val in enumerate(vals):
            self.cache[key][val_idx] = val
        return self.cache[key][idx]
