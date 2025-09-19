import os

import numpy as np
import json
import argparse

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mse, mae, mape

def import_data(diffusion_config, output_directory):
    local_path = "T{}_beta0{}_betaT{}_n{}".format(diffusion_config["T"],
                                              diffusion_config["beta_0"],
                                              diffusion_config["beta_T"],
                                              train_config['max_components'])

    # Get shared output_directory ready
    output_directory = os.path.join(output_directory, local_path)
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    file_types = ["original", "imputation", "mask"]


    results = {}
    for file_type in file_types:
        i=0
        results[file_type] = []
        file_path = os.path.join(output_directory, f"{file_type}{i}.npy")
        while os.path.exists(file_path):
            
            file_path = os.path.join(output_directory, f"{file_type}{i}.npy")
            if os.path.exists(file_path):
                print(file_path)
                data = np.load(file_path)
                results[file_type].append(data)
            else:
                print("File not found")
            i += 1
    
    return results,i
#107-190
def evaluate_results(results,j):
    total_mse = 0
    total_mae = 0
    total_mape = 0
    #print(i)
    #for i in range(len(results["original"])):
    for i in range(0,len(results["original"])):
        if i > 8 and i < 17:

            j-=1
        else:
            print("=====================================")
            print(results["original"][i].shape)
            print(np.median(results["imputation"][i],axis=0).shape)
            print(results["mask"][i].shape)
            print("=====================================")
            y_true = results["original"][i][results['mask'][i] ==0]
            y_pred = np.median(results["imputation"][i],axis=0)[results['mask'][i] ==0]
            mse, mae, mape = evaluate(y_true, y_pred)
            total_mse += mse
            total_mae += mae
            total_mape += mape
            print(f"mse: {mse}, mae: {mae}, mape: {mape}, {i} i")
            print("=====================================")
            print(i)
    
    return total_mse/(j), total_mae/(j), total_mape/(j)


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SSSDS4.json',
                        help='JSON file for configuration')

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    config = json.loads(data)
    print(config)

    train_config = config["train_config"]  # training parameters

    gen_config = config["gen_config"]  # to load generator

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters
 # dictionary of all diffusion hyperparameters
    print(gen_config['output_directory'])

    results,i = import_data(diffusion_config, gen_config["output_directory"])
    print(i)
    a,b,c = evaluate_results(results,i)

    print("final results")
    print(f"mse: {a}, mae: {b}, mape: {c}")


