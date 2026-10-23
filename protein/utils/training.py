"""Retained training updates, validation selection and graph-aware evaluation."""
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from protein.my_loader.graphs import loader, batches
from protein.utils.checkpoint import atomic, sha, save_weights, load_weights
from protein.utils.metrics import metric_dict
from protein.utils.model import make_model


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)

def make_optimizer(model, config, structure):
    geometric = [p for name, p in model.named_parameters() if name.startswith('geometry_')]
    ordinary = [p for name, p in model.named_parameters() if not name.startswith('geometry_')]
    return AdamW([
        dict(params=ordinary, lr=config['lr'], weight_decay=config['weight_decay']),
        dict(params=geometric, lr=config['lr'] * structure['lr_factor'], weight_decay=structure['weight_decay']),
    ])

def logits(model, batch, device):
    on = lambda value: None if value is None else value.to(device)
    return model(on(batch['esm']), on(batch['mask']), on(batch['chain_to_sample']), len(batch['labels']),
                 on(batch['hapr']), on(batch['pAPR']), on(batch['rAPR']),
                 graph={key: on(value) for key, value in batch['graph'].items()})

@torch.no_grad()
def evaluate(model, records, store, graphs, config, structure, arm, device):
    model.eval()
    labels, probabilities, ids = [], [], []
    loss = 0.
    for full in loader(records, store, graphs, arm != 'none', config['batch_size'], False, None):
        for batch in batches(full, config['microbatch_size'], structure):
            y = batch['labels'].to(device)
            prediction = logits(model, batch, device)
            assert torch.isfinite(prediction).all(), 'Nonfinite prediction'
            loss += float(nn.functional.binary_cross_entropy_with_logits(prediction, y, reduction='sum'))
            labels.append(y.cpu().numpy())
            probabilities.append(prediction.sigmoid().cpu().numpy())
            ids.extend(batch['sample_ids'])
    y, probability = np.concatenate(labels), np.concatenate(probabilities)
    assert ids == [r.sample_id for r in records]
    return {**metric_dict(y, probability), 'loss': loss / len(y)}, probability

def train_candidate(config, structure, records, store, graphs, device, fold, output_dir):
    arm = 'epoch20'
    job = {'esm': '650m', 'fold': fold}
    folder = Path(output_dir) / 'training'
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / 'validation.json'
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if Path(previous['checkpoint']).is_file():
            return previous
    weights = Path(output_dir) / 'checkpoints'
    weights.mkdir(parents=True, exist_ok=True)
    best_path, last_path = weights / 'best.pt.gz', weights / 'last.pt'
    base_seed = config['seed'] + 1009 * job['fold']
    seed(base_seed)
    model = make_model(config, structure, store, arm).to(device)
    optimizer = make_optimizer(model, config, structure)
    positive = sum(r.label for r in records['train'])
    negative = len(records['train']) - positive
    weight = (negative / positive) ** config['pos_weight_power']
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    generator = torch.Generator()
    training = loader(records['train'], store, graphs, arm != 'none', config['batch_size'], True, generator)
    history, best_auc, best_loss, best_epoch, stale, first = [], -math.inf, math.inf, 0, 0, 1
    if last_path.exists():
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        assert state['config'] == config and state['structure'] == structure
        model.load_state_dict(state['state_dict'])
        optimizer.load_state_dict(state['optimizer'])
        history, best_auc, best_loss = state['history'], state['best_auc'], state['best_loss']
        best_epoch, stale, first = state['best_epoch'], state['stale'], state['epoch'] + 1
        del state
    print(json.dumps(dict(event='candidate_start', job=job, arm=arm, next_epoch=first,
                          parameters=sum(p.numel() for p in model.parameters()), pos_weight=weight,
                          structure=structure)), flush=True)
    for epoch in range(first, config['max_epochs'] + 1):
        if stale >= config['patience']:
            break
        epoch_seed = base_seed + 100000 * epoch
        seed(epoch_seed)
        generator.manual_seed(epoch_seed + 73471)
        started = time.time()
        model.train()
        training_loss = 0.
        for full in training:
            optimizer.zero_grad(set_to_none=True)
            for batch in batches(full, config['microbatch_size'], structure):
                loss = criterion(logits(model, batch, device), batch['labels'].to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                (loss * (len(batch['labels']) / len(full['labels']))).backward()
                training_loss += float(loss.detach()) * len(batch['labels'])
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise FloatingPointError('Nonfinite gradient')
            optimizer.step()
        current, probability = evaluate(model, records['val'], store, graphs, config, structure, arm, device)
        current.update(epoch=epoch, train_loss=training_loss / len(records['train']), epoch_seconds=time.time() - started)
        history.append(current)
        improved = current['roc_auc'] > best_auc + 1e-5 or (abs(current['roc_auc'] - best_auc) <= 1e-5 and current['loss'] < best_loss)
        if improved:
            best_auc, best_loss, best_epoch, stale = current['roc_auc'], current['loss'], epoch, 0
            save_weights(best_path, dict(state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                config=config, structure=structure, arm=arm, esm=job['esm'], best_epoch=epoch,
                validation_probability=probability, validation_auroc=best_auc,
                validation_sample_ids=[r.sample_id for r in records['val']]))
        else:
            stale += 1
        temporary = last_path.with_name(last_path.name + '.tmp')
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), history=history,
                        best_auc=best_auc, best_loss=best_loss, best_epoch=best_epoch, stale=stale,
                        epoch=epoch, config=config, structure=structure), temporary)
        temporary.replace(last_path)
        atomic(folder / 'history.json', history)
        atomic(Path(output_dir) / 'status.json', dict(job=job, arm=arm, epoch=epoch, best_epoch=best_epoch,
                                                best_validation_auroc=best_auc, time=time.time()))
        print(json.dumps(dict(event='epoch', arm=arm, **current)), flush=True)
    state = load_weights(best_path)
    model.load_state_dict(state['state_dict'])
    selected, probability = evaluate(model, records['val'], store, graphs, config, structure, arm, device)
    error = float(np.max(np.abs(probability - state['validation_probability'])))
    assert error <= 1e-5, 'Checkpoint roundtrip changed validation probabilities'
    result = dict(validation=selected, best_epoch=best_epoch, pos_weight=weight, pos_weight_power=config['pos_weight_power'],
                  train_positive=positive, train_negative=negative, train_n=len(records['train']), val_n=len(records['val']),
                  epochs_run=len(history), parameter_count=sum(p.numel() for p in model.parameters()),
                  checkpoint=str(best_path), checkpoint_probability_error=error)
    atomic(result_path, result)
    last_path.unlink(missing_ok=True)
    del model, optimizer, state
    torch.cuda.empty_cache()
    return result
