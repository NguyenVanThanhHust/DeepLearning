import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(train.values)

dir(pd)

train.category_name = train.category_name.fillna('missing').astype('category')
train.brand_name = train.brand_name.fillna('missing').astype('category')
train.item_condition_id = train.item_condition_id.astype('category')
test.category_name = test.category_name.fillna('missing').astype('category')
test.brand_name = test.brand_name.fillna('missing').astype('category')
test.item_condition_id = test.item_condition_id.astype('category')