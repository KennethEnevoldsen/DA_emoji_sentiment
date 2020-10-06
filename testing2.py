"""
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from transformers import BertConfig
from transformers import TFBertForSequenceClassification
from transformers import TFBertModel, AutoTokenizer

config = BertConfig.from_pretrained("bert-base-multilingual-cased")
max_len = config.max_position_embeddings
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")


def return_id(str1, str2, length=max_len):

    inputs = tokenizer.encode_plus(str1, str2,
                                   add_special_tokens=True,
                                   max_length=length)

    input_ids = inputs["input_ids"]
    input_masks = [1] * len(input_ids)
    input_segments = inputs["token_type_ids"]

    padding_length = length - len(input_ids)
    padding_id = tokenizer.pad_token_id

    input_ids = input_ids + ([padding_id] * padding_length)
    input_masks = input_masks + ([0] * padding_length)
    input_segments = input_segments + ([0] * padding_length)

    return [input_ids, input_masks, input_segments]


# encode 3 sentences
input_ids, input_masks, input_segments = [], [], []
for instance in ['hello hello', 'ciao ciao', 'marco marco']:

    ids, masks, segments = \
        return_id(instance, None)

    input_ids.append(ids)
    input_masks.append(masks)
    input_segments.append(segments)

input_ = [np.asarray(input_ids, dtype=np.int32),
          np.asarray(input_masks, dtype=np.int32),
          np.asarray(input_segments, dtype=np.int32)]


input_

input_layer = tf.keras.Input(shape=(512,), dtype='int64')
bert = TFBertModel.from_pretrained("bert-base-multilingual-cased")(input_layer)
bert = bert[0]
model = tf.keras.Model(inputs=input_layer, outputs=bert)
model.predict(np.asarray(input_ids, dtype=np.int32))

bert = TFBertModel.from_pretrained("bert-base-multilingual-cased")
resd = bert((np.asarray(input_ids, dtype=np.int32)), return_dict=True)
resd["pooler_output"]
res = bert((np.asarray(input_ids, dtype=np.int32)))
res[1]
res = tokenizer.encode("this is a test", padding="max_length",
                       add_special_tokens=True, truncation=True, return_tensors="tf")
len(res)
res