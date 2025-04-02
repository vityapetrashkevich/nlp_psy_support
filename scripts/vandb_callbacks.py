import time
import psutil
import torch
import wandb
from transformers import TrainerCallback


# Кастомный callback для логирования метрик
class WandbMetricsCallback(TrainerCallback):
    def __init__(self):
        self.epoch_start_time = None
        self.step_start_time = None
        self.train_start_time = None

    # Начало обучения
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.train_start_time = time.time()
        # Логируем количество обучаемых параметров
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        wandb.log({"num_trainable_params": num_params})

    # Начало эпохи
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    # Конец эпохи
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        wandb.log({"epoch_time": epoch_time, "epoch": state.epoch})

    # Начало шага
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()

    # Конец шага
    def on_step_end(self, args, state, control, model=None, **kwargs):
        step_time = time.time() - self.step_start_time
        wandb.log({"step_time": step_time, "global_step": state.global_step})

        # Логируем ресурсы
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        wandb.log({"cpu_percent": cpu_percent, "memory_percent": memory.percent})

        # Логируем норму градиента (каждые logging_steps)
        if state.global_step % args.logging_steps == 0 and model is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            wandb.log({"grad_norm": grad_norm.item()})

        # Логируем скорость обучения
        lr = args.learning_rate if not hasattr(state, 'optimizer') else state.optimizer.param_groups[0]['lr']
        wandb.log({"learning_rate": lr})

    # Логирование стандартных метрик и перплексии
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            wandb.log(logs)  # Логируем train_loss, eval_loss и другие стандартные метрики
            if "eval_loss" in logs:
                perplexity = torch.exp(torch.tensor(logs["eval_loss"]))
                wandb.log({"eval_perplexity": perplexity.item()})

    # Конец обучения
    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time
        wandb.log({"total_training_time": total_time})

