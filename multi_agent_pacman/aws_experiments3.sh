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
touch $data_path'/proximity_agent.csv'

wait
echo "All data files created"

# Run proximity system
(python2 pacman.py -p ProximityAgent -n $n_games -l smallClassic -g DirectionalGhost -q --fname $data_path'/proximity_agent.csv';
echo ">>>Proximity Agent - Complete<<<")&

wait

echo "All done"
