import re
import numpy as np
import os


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


# AF:0 HB:1 SP:2
# neg:0 neu:1 pos:2
# ex_pos:9 ve_pos:8 pos:7 sli_pos:6 neu:5 sli_neg:4 neg:3 ve_neg:2 ex_neg:1
def get_feature(PATH):
    training_features = None
    training_labels = []
    for f in os.listdir(PATH):
        file_path = os.path.join(PATH, f)
        #l = [0] * 9
        print(file_path)
        if "negative" in f or "arm" in f or '1' in f:
            #l[0] = l[0] + 1
            #if l[0] > 288:
            #    continue
            if training_features is None:
                training_features = np.load(file_path)
                labels = [0 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [0 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_labels.extend(labels)

        if "neutral" in f or "head" in f or '2' in f:
            #l[1] = l[1] + 1
            #if l[1] > 288:
            #    continue
            if training_features is None:
                training_features = np.load(file_path)
                labels = [1 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [1 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_labels.extend(labels)

        if "positive" in f or "spin" in f or '3' in f:
            #l[2] = l[2] + 1
            #if l[2] > 288:
            #    continue
            if training_features is None:
                training_features = np.load(file_path)
                labels = [2 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [2 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_labels.extend(labels)

        if '4' in f:
            #l[3] = l[3] + 1
            #if l[3] > 288:
            #    continue
            if training_features is None:
                training_features = np.load(file_path)
                labels = [3 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [3 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_labels.extend(labels)

        if '5' in f:
            #l[4] = l[4] + 1
            #if l[4] > 288:
            #    continue
            if training_features is None:
                training_features = np.load(file_path)
                labels = [4 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [4 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_labels.extend(labels)

        if '6' in f:
            #l[5] = l[5] + 1
            #if l[5] > 288:
            #    continue
            if training_features is None:
                training_features = np.load(file_path)
                labels = [5 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [5 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_labels.extend(labels)

        if '7' in f:
            #l[6] = l[6] + 1
            #if l[6] > 288:
            #    continue
            if training_features is None:
                training_features = np.load(file_path)
                labels = [6 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [6 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_labels.extend(labels)

        if '8' in f:
            #l[7] = l[7] + 1
            #if l[7] > 288:
            #    continue
            if training_features is None:
                training_features = np.load(file_path)
                labels = [7 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [7 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_labels.extend(labels)

        if '9' in f:
            #l[8] = l[8] + 1
            #if l[8] > 288:
            #    continue
            if training_features is None:
                training_features = np.load(file_path)
                labels = [8 for i in range(len(training_features))]
            else:
                features = np.load(file_path)
                labels = [8 for i in range(len(features))]
                training_features = np.concatenate((training_features, features), axis=0)
            training_labels.extend(labels)
    print(training_features.shape)
    return training_features, np.array(training_labels).squeeze()
