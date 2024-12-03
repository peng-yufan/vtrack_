
## Install the environment

```
conda env create -f environment.yml
```

## Data Preparation
Put the tracking datasets in ./data. It should look like:
   ```
   ${ROOT}
    -- data
        -- lasot
            |-- airplane
            |-- basketball
            |-- bear
            ...
        -- got10k
            |-- test
            |-- train
            |-- val
        -- trackingnet
            |-- TRAIN_0
            |-- TRAIN_1
            ...
            |-- TRAIN_11
            |-- TEST
   ```
## Set project paths

Run the following command to set paths for this project

```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .
```
After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Train 

```

python tracking/train.py --script mynet --config baseline --save_dir . --mode single --script_prv cttrak --config_prv baseline  
python tracking/train.py --script mynet --config baseline_large --save_dir . --mode single --script_prv cttrak --config_prv baseline_large  

```

## Test 


- UAV123
```
python tracking/test.py videotrack baseline-test-4.5-100-200-sigmoid3.0-10-3-5-2 --dataset uav --threads 32
python tracking/test.py videotrack baseline_large-test-4.7-60-200-sigmoid1.8-10-3-5-2 --dataset uav --threads 32


```
- LaSOT
```
python tracking/test.py videotrack SUCCESS-baseline-240-700-sigmoid3.0--dataset lasot --threads 32
python tracking/test.py videotrack baseline_large-test-4.9-150-700-sigmoid1.5 --dataset lasot --threads 32
```
- GOT10K-test
```
python tracking/test.py videotrack baseline-test-4.5-20-200-sigmoid3.0-10-3-5-2 --dataset got10k_test --threads 32
python tracking/test.py videotrack baseline_large-test-4.3-20-200-sigmoid1.8-10-3-5-2 --dataset got10k_test --threads 32
```
- TrackingNet 
```
python tracking/test.py videotrack baseline-test-4.0-20-200-sigmoid3.0-10-3-5-2 --dataset trackingnet --threads 32
python tracking/test.py videotrack baseline_large-test-4.0-20-200-sigmoid1.8-10-3-5-2 --dataset trackingnet --threads 32

```


### Evaluate 

**LaSOT/GOT10k-test/TrackingNet/OTB100/UAV123**

```
python tracking/analysis_results.py {script}  {config}  {dataset_name}
```

**For example**

```
python tracking/analysis_results.py mynet baseline trackingnet
```

**Pack TrackingNet**
```
#python lib/test/utils/transform_trackingnet.py --tracker_name videotrack --cfg_name {submitted baseline}
```

**Pack Got10k**
```
##python lib/test/utils/transform_got10k.py --tracker_name videotrack --cfg_name {submitted baseline}
```

