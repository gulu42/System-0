#!/bin/bash

# Parameters
probs=(0.1 0.3 0.5 0.7 0.9)
food_thresh=(0.1 0.3 0.5 0.7 0.9)
n_games=1000
data_path='aws_data'

# parameters for mini sample run
# n_games=3
# data_path='mini_aws_data'

# Create all data files
touch $data_path'/system_2.csv'
touch $data_path'/system_1.csv'

wait
echo "All data files created"

# Set system 2 baseline
(python2 pacman.py -p System2Agent -n $n_games -l smallClassic -g DirectionalGhost -q --fname $data_path/system_2.csv;
echo ">>>System 2 - Complete<<<")&

# Set system 1 baseline
(python2 pacman.py -p System1Agent -n $n_games -l smallClassic -g DirectionalGhost -q --fname $data_path/system_1.csv;
echo ">>>System 1 - Complete<<<")&

wait

echo "All done"
