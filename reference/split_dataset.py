from landmark_detection_dataset import LandmarkDetectionDataset
import random

def split_dataset(dataset, ratio=1, shuffle=True):
    if ratio >= 1:
        return dataset, None
    elif ratio <= 0:
        return None, dataset
    
    total_samples = {}
    for i in range(len(dataset.labels)):
        category_id = dataset.labels[i][1]
        if category_id in total_samples:
            total_samples[category_id] += 1
        else:
            total_samples[category_id] = 1

    set_a_samples = {}
    set_b_samples = {}
    for category_id in list(total_samples.keys()):
        set_a_samples[category_id] = round(total_samples[category_id] * ratio)
        set_b_samples[category_id] = total_samples[category_id] - set_a_samples[category_id]

        if set_a_samples[category_id] <= 0 and set_b_samples[category_id] >= 2:
            set_a_samples[category_id] += 1
            set_b_samples[category_id] -= 1
        elif set_b_samples[category_id] <= 0 and set_a_samples[category_id] >= 2:
            set_b_samples[category_id] += 1
            set_a_samples[category_id] -= 1

    if shuffle:
        random.shuffle(dataset.labels)

    set_a_labels = []
    set_b_labels = []
    for i in range(len(dataset.labels)):
        label = dataset.labels[i]

        category_id = label[1]
        if set_a_samples[category_id] > 0:
            set_a_labels.append(label)
            set_a_samples[category_id] -= 1
        else:
            set_b_labels.append(label)
            set_b_samples[category_id] -= 1

    dataset_a = LandmarkDetectionDataset(dataset.data_dir, None, dataset.num_classes)
    dataset_b = LandmarkDetectionDataset(dataset.data_dir, None, dataset.num_classes)

    dataset_a.labels = set_a_labels
    dataset_b.labels = set_b_labels

    return dataset_a, dataset_b
