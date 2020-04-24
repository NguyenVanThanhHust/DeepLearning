# Open Dataset
import os

data_annotation_file = "../../Dataset/Flickr8k/Flickr8k_text/Flickr8k.token.txt"
file = open(data_annotation_file)
content = file.read()

descriptions = dict()
for line in doc.split('\n'):
    # split line by white space
    tokens = line.split()
    
    # first token is image id, the rest are descriptions
    img_id, img_desc = tokens[0], tokens[1:]
    
    # extract filename from image id
    img_id = img_id.split('.')[0]
    
    # convert descriptions tokens back to string
    img_desc = ' '.join(img_desc)
    if img_id not in descriptions:
        descriptions[img_id] = list()
    descriptions[img_id].append(img_desc)
    
    
# Clean data 
table = str.maketrans('', '', string.punctuation)
for key, desc_list in descriptions.items():
    for i in range(len(desc_list)):
        desc = desc_list[i]
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(table) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word)>1]
        # remove tokens with numbers in them
        desc = [word for word in desc if word.isalpha()]
        # store as string
        desc_list[i] =  ' '.join(desc)
        
vocabulary = set()
for key in descriptions.keys():
    [vocabulary.update(d.split()) for d in descriptions[key]]
print('Original Vocabulary Size: %d' %len(vocabulary))

# Consider those words which occur at least 10 times in the entire corpus
all_train_captions = []
for key, val in train_des