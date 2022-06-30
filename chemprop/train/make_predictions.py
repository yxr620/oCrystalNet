from argparse import Namespace
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pymatgen.io.vasp import Poscar

from .predict import predict
from chemprop.data import CrystalDataset, CrystalDatapoint
from chemprop.utils import load_args, load_checkpoint, load_scalers, get_metric_func
from chemprop.features import AtomCustomJSONInitializer, GaussianDistance, load_radius_dict
from chemprop.data.utils import get_datapoint, get_task_names
from .evaluate import evaluate, evaluate_predictions

def make_predictions(args: Namespace):
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :return: A list of lists of target name and predictions.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    test_name = pd.read_csv(f'{args.test_path}/{args.split_dir}/test.csv')['name'].tolist()
    # test_name = list(sorted(test_graph_dict.keys(), key=lambda x: int(x.split('-')[1])))

    print('Loading data')
    with open(f'{args.test_path}/{args.graph_dict}', 'rb') as f:
        all_graph = pickle.load(f)

    # print(all_graph['POSCAR-JVASP-21210'])
    # exit()
    test_data = get_datapoint(path=f'{args.test_path}/{args.split_dir}/test.csv', all_graph=all_graph, args=args)

    print(f'Test size = {len(test_data):,}')

#-----------------------------

    task_indices = get_task_names(path=f'{args.test_path}/{args.property}', use_compound_names=True)
    task_index = task_indices[args.dataset_name]

    test_targets = [[targets[task_index]] for targets in test_data.targets()]
    test_data.set_targets(test_targets)

    # Predict with each model individually and sum predictions
    if args.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), args.num_tasks))

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
        fold_num = checkpoint_path.split('/')[-3].split('_')[-1]
        model = load_checkpoint(checkpoint_path, cuda=args.cuda)
        model_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        sum_preds += np.array(model_preds)
        # write fold prediction
        with open(f'{args.test_path}/{args.split_dir}/predict_{args.dataset_name}_fold_{fold_num}_crystalnet.csv', 'w') as fw:
            fw.write(f'name,{args.dataset_name}\n')

            for name, prediction in zip(test_name, np.array(model_preds)):
                fw.write(f'{name},{",".join([str(pre) for pre in prediction])}\n')

    # Ensemble predictions
    avg_preds = sum_preds / len(args.checkpoint_paths)
    avg_preds = avg_preds.tolist()

    mae_score = evaluate_predictions(
        preds=avg_preds,
        targets=test_data.targets(),
        num_tasks=args.num_tasks,
        metric_func=get_metric_func('mae'),
        dataset_type=args.dataset_type
    )

    r2_score = evaluate_predictions(
        preds=avg_preds,
        targets=test_data.targets(),
        num_tasks=args.num_tasks,
        metric_func=get_metric_func('r2'),
        dataset_type=args.dataset_type
    )
    print(f'{args.test_path}/{args.split_dir}/predict_{args.dataset_name}_fold_{fold_num}_crystalnet mae: {mae_score}')
    print(f'{args.test_path}/{args.split_dir}/predict_{args.dataset_name}_fold_{fold_num}_crystalnet r2: {r2_score}')


    return test_name, avg_preds





