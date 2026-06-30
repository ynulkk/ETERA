import os
import re
import json
import glob
import random
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch

PATH_SORTING = "natural_sorted_v1"


def natural_sort_key(value):
    """Return a natural-sort key so trial_2 sorts before trial_10."""
    text = str(value).replace("\\", "/")
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def natural_sorted(values):
    return sorted(values, key=natural_sort_key)


def _norm_path(path):
    return str(path).replace("\\", "/")


def list_paths_sorted(path, pattern="*", recursive=False, files_only=False, dirs_only=False):
    """List paths with deterministic natural sorting.

    Use this instead of os.listdir/glob.glob/Path.glob in dataset loaders.
    """
    base = Path(path)
    iterator = base.rglob(pattern) if recursive else base.glob(pattern)
    paths = []
    for item in iterator:
        if files_only and not item.is_file():
            continue
        if dirs_only and not item.is_dir():
            continue
        paths.append(_norm_path(item))
    return natural_sorted(paths)


def list_files_sorted(path, suffix=None, pattern="*", recursive=False):
    files = list_paths_sorted(path, pattern=pattern, recursive=recursive, files_only=True)
    if suffix is None:
        return files
    suffixes = (suffix,) if isinstance(suffix, str) else tuple(suffix)
    return [p for p in files if p.endswith(suffixes)]


def glob_sorted(pattern, recursive=False):
    return natural_sorted(_norm_path(p) for p in glob.glob(pattern, recursive=recursive))


def _collect_npy_files(path: str):
    return list_files_sorted(path, suffix=".npy", recursive=True)


def get_train_test_filets_path(data_path: str, num_trial: int, test_sessionId=-1):
    train_npy_path_list = []
    test_npy_path_list = []
    for i in range(15):
        subject_path = os.path.join(data_path, f'subject_{i + 1}').replace('\\', '/')
        for j in range(3):
            session_path = os.path.join(subject_path, f'session_{j + 1}').replace('\\', '/')
            session_files = _collect_npy_files(session_path)
            if j == test_sessionId:
                test_npy_path_list.extend(session_files)
            else:
                train_npy_path_list.extend(session_files)
    return train_npy_path_list, test_npy_path_list


def get_subject_train_test_filets_path(data_path: str, num_trial: int, test_subjectId: int, test_sessionId=3):
    assert test_subjectId <= 15 and test_subjectId >= 1, "test_subjectId must be between 1 and 15"
    assert test_sessionId <= 3 and test_sessionId >= 1, "test_sessionId must be between 1 and 3"
    session_path = os.path.join(data_path, f'subject_{test_subjectId}/session_{test_sessionId}').replace('\\', '/')
    return _collect_npy_files(session_path)


def subject_dependent_files_path(data_path: str, num_train: int, num_trial: int, test_subjectId: int, test_sessionId=3):
    assert test_subjectId <= 15 and test_subjectId >= 1, "test_subjectId must be between 1 and 15"
    assert test_sessionId <= 3 and test_sessionId >= 1, "test_sessionId must be between 1 and 3"
    train_path_list, test_path_list = [], []
    session_path = os.path.join(data_path, f'subject_{test_subjectId}/session_{test_sessionId}').replace('\\', '/')
    for file_path in _collect_npy_files(session_path):
        sample_id = int(os.path.splitext(file_path)[0].split('_')[-2])
        if sample_id <= num_train:
            train_path_list.append(file_path)
        else:
            test_path_list.append(file_path)
    return train_path_list, test_path_list


def get_subject_train_filets_path(data_path: str, subjectId: int):
    assert subjectId <= 15 and subjectId >= 1, "subjectId must be between 1 and 15"
    train_npy_path_list = []
    for sessionId in range(1, 4):
        session_path = os.path.join(data_path, f'subject_{subjectId}/session_{sessionId}').replace('\\', '/')
        train_npy_path_list.extend(_collect_npy_files(session_path))
    return train_npy_path_list


def get_cross_subject_train_test_filets_path(data_path: str, num_trial: int, test_subjectId=1, sessionId=-1):
    assert sessionId <= 3 and sessionId >= 1, "test_experimentID must be between 1 and 3"
    assert test_subjectId <= 15 and test_subjectId >= 1, "test_subjectId must be between 1 and 15"
    train_npy_path_list = []
    test_npy_path_list = []
    for i in range(15):
        session_path = os.path.join(data_path, f'subject_{i + 1}/session_{sessionId}').replace('\\', '/')
        session_files = _collect_npy_files(session_path)
        if (i + 1) == test_subjectId:
            test_npy_path_list.extend(session_files)
        else:
            train_npy_path_list.extend(session_files)
    return train_npy_path_list, test_npy_path_list


def get_cross_subject_all_session_filets_path(data_path: str, num_trial: int, test_subjectId=1):
    assert test_subjectId <= 15 and test_subjectId >= 1, "test_subjectId must be between 1 and 15"
    train_npy_path_list = []
    test_npy_path_list = []
    for i in range(15):
        for sessionId in range(1, 4):
            session_path = os.path.join(data_path, f'subject_{i + 1}/session_{sessionId}').replace('\\', '/')
            session_files = _collect_npy_files(session_path)
            if (i + 1) == test_subjectId:
                test_npy_path_list.extend(session_files)
            else:
                train_npy_path_list.extend(session_files)
    return train_npy_path_list, test_npy_path_list


def get_cross_session_train_test_filets_path(data_path: str, train_session=-11, test_session=-1):
    assert train_session <= 3 and train_session >= 1, "train_session must be between 1 and 3"
    assert test_session <= 3 and test_session >= 1, "test_session must be between 1 and 3"
    train_npy_path_list = []
    test_npy_path_list = []
    for i in range(15):
        subject_path = os.path.join(data_path, f'subject_{i + 1}').replace('\\', '/')
        for j in range(3):
            session_path = os.path.join(subject_path, f'session_{j + 1}').replace('\\', '/')
            session_files = _collect_npy_files(session_path)
            if (j + 1) == train_session:
                train_npy_path_list.extend(session_files)
            elif (j + 1) == test_session:
                test_npy_path_list.extend(session_files)
    return train_npy_path_list, test_npy_path_list


def get_subject_cross_session_train_test_filets_path(data_path: str, subjectID: int, num_trial: int, train_session=-11, test_session=-1):
    assert train_session <= 3 and train_session >= 1, "train_session must be between 1 and 3"
    assert test_session <= 3 and test_session >= 1, "test_session must be between 1 and 3"
    train_npy_path_list = []
    test_npy_path_list = []
    subject_path = os.path.join(data_path, f'subject_{subjectID}').replace('\\', '/')
    for j in range(1, 4):
        session_path = os.path.join(subject_path, f'session_{j}').replace('\\', '/')
        session_files = _collect_npy_files(session_path)
        if j == train_session:
            train_npy_path_list.extend(session_files)
        elif j == test_session:
            test_npy_path_list.extend(session_files)
    return train_npy_path_list, test_npy_path_list


def _label_from_path(path_name):
    return int(os.path.splitext(path_name)[0].split('_')[-1])


def _read_split_manifest(manifest_path):
    payload = json.loads(Path(manifest_path).read_text())
    support = payload.get("support_files") or payload.get("support") or payload.get("target_support")
    query = payload.get("query_files") or payload.get("query") or payload.get("test") or payload.get("query_test")
    if support is None or query is None:
        raise ValueError(f"Manifest {manifest_path} does not contain support/query split keys")
    return [_norm_path(p) for p in support], [_norm_path(p) for p in query], payload


def cross_subject_n_shot(test_path_list, num_shot, num_classes, seed=None, manifest_path=None, manifest_metadata=None, dataset=None, session=None, test_subject=None):
    """Create or reuse a deterministic N-shot support/query split.

    If manifest_path exists, support/query are loaded from it. Otherwise the input
    paths are natural-sorted, sampled with a local fixed-seed RNG, and saved as JSON.
    Backward compatible: with no manifest_path and no seed, it uses the global random
    module after natural sorting.
    """
    if manifest_path and Path(manifest_path).exists():
        support_set, query_set, _ = _read_split_manifest(manifest_path)
        return support_set, query_set

    sorted_paths = natural_sorted(_norm_path(p) for p in test_path_list)
    class_label_list = [[] for _ in range(num_classes)]
    for path_name in sorted_paths:
        class_label_list[_label_from_path(path_name)].append(path_name)

    rng = random.Random(int(seed)) if seed is not None else random
    support_set, query_set = [], []
    for class_label, samples in enumerate(class_label_list):
        if len(samples) < num_shot:
            raise ValueError(f"class {class_label} has {len(samples)} samples < num_shot={num_shot}")
        support_samples = rng.sample(samples, num_shot)
        support_lookup = set(support_samples)
        query_samples = [sample for sample in samples if sample not in support_lookup]
        support_set.extend(natural_sorted(support_samples))
        query_set.extend(query_samples)

    if manifest_path:
        meta = dict(manifest_metadata or {})
        payload = {
            **meta,
            "dataset": dataset if dataset is not None else meta.get("dataset"),
            "session": session if session is not None else meta.get("session"),
            "test_subject": test_subject if test_subject is not None else meta.get("test_subject"),
            "shot": num_shot,
            "seed": int(seed) if seed is not None else None,
            "num_classes": num_classes,
            "support_files": support_set,
            "query_files": query_set,
            "path_sorting": PATH_SORTING,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest = Path(manifest_path)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return support_set, query_set


def seed_everything(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    return seed


def seed_torch(seed=42):
    return seed_everything(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed=42):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def make_worker_init_fn(seed=42):
    base_seed = int(seed)
    def _init_fn(worker_id):
        worker_seed = (base_seed + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    return _init_fn
