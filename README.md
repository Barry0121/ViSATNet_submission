# ViSATNet: Visual SATNet for Image Classification

ViSATNet uses SATNet to learn relational rules between pixels to perform classification and tasks.

# Setups

#### Using `pip`:

1. Install required packages.

   `pip3 install -r requirements.txt`

2. Install SATNet from source.

   `python3 installation/setup_satnet.py install`

#### Using `conda`:

1. Install required packages in the environment `satnet_env`.

   `conda env create -f environment.yml`

2. Activate the environment.

   `conda activate satnet_env`

3. Install SATNet from source.

   `python3 installation/setup_satnet.py install`

# Training the model

Training a hierarchical model (2 stacks of SATNet) for classification.

### Example

```
python3 mnist_rule/train_eval.py --loss ce --model hrclf --epoch 30
```

You can find the model weight and output are under `output/[dataset]/[model_name]`.

# Extract Configuration and Visualization

We use the script `mnist_rule/generate_dist.py` to extract the hidden features of SATNet inside of ViSATNet.

You can do the following by specifying the argument for `--mode`:

- Generate binary configuration distribution over the testing set. (`config_distribution`)

- Generate float bits distribution over images with different target
  labels and across the testing set. (`bits_distribution`)

- Visualize each configuration's patches, averaged over all the corresponding instances. (`visualize_patches`)

- Select an image from the testing set and generate new images by switching patches according to their configurations. (`verify_patches`)

- Extract and save the float bits in a pickle file. (`extract_dataset`)

### Example

```
python3 mnist_rule/generate_dist.py --dataset mnist --mode verify_patches --stride 7 --hidden_dim 8 --aux 50
```

All the results are stored under `output/[dataset]/[model_name]/[digits]`.
