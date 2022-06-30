'''
nohup python -u predict.py --gpu 0 --seed 0 --data_path ./data/material_project --test_path ./data/material_project --dataset_name band_gap --checkpoint_dir ./ckpt/ensemble_band_gap/ --no_features_scaling > ./log/predict_band_gap.log 2>&1 &
'''


from chemprop.parsing import parse_predict_args, modify_predict_args
from chemprop.train import make_predictions
from chemprop.train.evaluate import evaluate, evaluate_predictions

if __name__ == '__main__':
    args = parse_predict_args()
    print(args)
    test_name, test_prediction = make_predictions(args)
    # print(test_name)
    # print(test_prediction.size())

    exit()
    with open(f'{args.test_path}/seed_{args.seed}/predict_{args.dataset_name}_crystalnet.csv', 'w') as fw:
        fw.write(f'name,{args.dataset_name}\n')

        for name, prediction in zip(test_name, test_prediction):
            fw.write(f'{name},{",".join([str(predict) for predict in prediction])}\n')
