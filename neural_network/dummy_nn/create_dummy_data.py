import pandas as pd
import numpy as np
import random

lower_limit = 1
upper_limit = 1000
num_entries = 1000

def foo_linear(x1,x2,x3): #func to be approximated
    return x1 + x2 + x3

def foo_square(x1,x2,x3): #func to be approximated
    return x1 + (x2 ^ 2) + x3

df = pd.DataFrame(columns=['x1','x2','x3','label'])

for i in range(num_entries):
    x1 = random.uniform(lower_limit,upper_limit)
    x2 = random.uniform(lower_limit,upper_limit)
    x3 = random.uniform(lower_limit,upper_limit)
    temp_dict = {'x1':x1,'x2':x2,'x3':x3}
    temp_dict['label'] = foo_linear(x1,x2,x3)
    # df.append(pd.Series([x1,x2,x3,foo_linear(x1,x2,x3)],index = df.columns),ignore_index = True)
    df = df.append(temp_dict,ignore_index = True)

df.to_csv('linear_train_data.csv', index = False)
# df.to_csv('linear_train_data.csv', index = False, header = None)
