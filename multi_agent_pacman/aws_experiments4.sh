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

for i in "${food_thresh[@]}"; do
    touch $data_path'/food_thresh_'$i'.csv'
done

wait
echo "All data files created"

# Run proximity plus food density system
for i in "${food_thresh[@]}"; do
    (echo "Running food thresh "$i;
    python2 pacman.py -p ProximityAndFoodAgent -n $n_games -l smallClassic -g DirectionalGhost -q -a food_thresh=$i --fname $data_path'/food_thresh_'$i'.csv';
    echo '>>Proximity plus food ('$i') - Complete<<')&
done

wait

echo "All done"
