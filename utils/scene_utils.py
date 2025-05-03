def load_scene_metadata(path='data/categories_places365.txt'):
    class_to_label = {}
    label_to_inout = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_name = '_'.join(parts[:-2])
            class_idx = int(parts[-2])
            inout = parts[-1]
            class_to_label[class_name] = class_idx
            label_to_inout[class_idx] = inout
    return class_to_label, label_to_inout