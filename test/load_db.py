import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tqdm import tqdm
from images_framework.src.datasets import Database

def load_annotations(anns_file):
    """
    Load ground truth annotations according to each database.
    """
    print('Open annotations file: ' + str(anns_file))
    if os.path.isfile(anns_file):
        pos = anns_file.rfind('/') + 1
        path = anns_file[:pos]
        file = anns_file[pos:]
        db = file[:file.find('_ann')]
        datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
        with open(anns_file, 'r', encoding='utf-8') as ifs:
            lines = ifs.readlines()
            anns = []
            for i in tqdm(range(len(lines)), file=sys.stdout):
                parts = lines[i].strip().split(',')
                if parts[0] == '@':
                    db = parts[1]
                if parts[0] == '#' or parts[0] == '@':
                    continue
                idx = next((idx for idx, subset in enumerate(datasets) if db in subset), None)
                if idx is None:
                    raise ValueError('Database does not exist')
                seq = Database.__subclasses__()[idx]().load_filename(path, db, lines[i])
                if len(seq.images) == 0:
                    continue
                anns.append(seq)
        ifs.close()
    else:
        raise ValueError('Annotations file does not exist')
    return anns

load_annotations("./data/pcr-op3d-12p/image_data.txt")