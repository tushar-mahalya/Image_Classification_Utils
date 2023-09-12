def get_logger(filename):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler2)
    return logger

# This is a function necessary to achieve reproducibility.
def worker_init_fn(worker_id):
    torch.manual_seed(worker_id)
    random.seed(worker_id)
    np.random.seed(worker_id)
    torch.cuda.manual_seed(worker_id)
    os.environ['PYTHONHASHSEED'] = str(worker_id)

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_folds(data, num_splits, seed):
    data["kfold"] = -1

    mskf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    labels = ["label"]
    data_labels = data[labels].values

    for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
        data.loc[v_, "kfold"] = f

    return data
