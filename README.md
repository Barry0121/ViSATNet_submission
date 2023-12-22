# satnet_image_completion

Using satnet to learn relational rules between pixels in order to perform both classification and generative tasks

# Setups

-- required packages

`pip3 install -r requirements.txt`

-- install SATNet

`python3 installation/setup_satnet.py install`

# to run the training

-- training a hierarchical model (2 stacks of satnet) for classification

`python3 mnist_rule/train_eval.py --loss ce --model hrclf --epoch 30`
