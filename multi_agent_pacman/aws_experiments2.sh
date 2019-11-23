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
for i in "${probs[@]}"; do
    touch $data_path'/random_choice_'$i'.csv'
done

wait
echo "All data files created"

# Run multiple random probs
for i in "${probs[@]}"; do
    (echo "Running prob "$i;
    python2 pacman.py -p RandomChoiceAgent -n $n_games -l smallClassic -g DirectionalGhost -q -a prob_sys1=$i --fname $data_path'/random_choice_'$i'.csv';
    echo '>>Random choice ('$i') - Complete<<')&
done

wait

echo "All done"
