import os.path
import numpy as np
import scipy.io as sio
import sklearn.preprocessing as skp
import torch
from scipy import sparse


def load_mat(args):
    data_X = []
    label_y = None

    if args.dataset == "LandUse21":
        mat = sio.loadmat(os.path.join(args.data_path, "LandUse_21.mat"))
        data_X.append(sparse.csr_matrix(mat["X"][0, 1]).toarray())
        data_X.append(sparse.csr_matrix(mat["X"][0, 2]).toarray())
        label_y = np.squeeze(mat["Y"]).astype("int")

    elif args.dataset == "Caltech101":
        mat = sio.loadmat(
            os.path.join(args.data_path, "2view-caltech101-8677sample.mat")
        )
        X = mat["X"][0]
        data_X.append(X[0].T)
        data_X.append(X[1].T)
        label_y = np.squeeze(mat["gt"]) - 1

    elif args.dataset == 'Animal':
        mat = sio.loadmat(
            os.path.join(args.data_path, 'AWA-7view-10158sample.mat')
        )
        label_y = np.squeeze(mat['gt'])
        data_X.append(mat['X'][0][6].T)
        data_X.append(mat['X'][0][5].T)

    elif args.dataset == "umpc_food101":
        mat = sio.loadmat(os.path.join(args.data_path, "umpc_food101.mat"))
        img_features = mat["img_features"].astype(np.float32)
        txt_features = mat["txt_features"].astype(np.float32)
        label_y = np.squeeze(mat["labels"])
        data_X.append(img_features)
        data_X.append(txt_features)

    elif args.dataset == "Scene15":
        mat = sio.loadmat(os.path.join(args.data_path, "Scene_15.mat"))
        X = mat["X"][0]
        data_X.append(X[1].astype("float32"))
        data_X.append(X[0].astype("float32"))
        label_y = np.squeeze(mat["Y"])

    elif args.dataset == "XMediaNet":
        mat = sio.loadmat(os.path.join(args.data_path, "xmedia_deep_2_view.mat"))
        data_X.append(mat['Img'])
        data_X.append(mat['Txt'])
        label_y = np.squeeze(mat['label'])
    else:
        raise KeyError(f"Unknown Dataset {args.dataset}")
    
    label_y = label_y - np.min(label_y)  # make sure label starts from 0

    raw_data = [data.copy() for data in data_X]
    raw_label = label_y.copy()

    # Control the randomness of the data
    rng = np.random.RandomState(1234)
    label_y_view2 = label_y.copy()
    id1 = np.arange(data_X[0].shape[0])
    id2 = id1.copy()
    is_align = np.ones_like(label_y)
    if args.m_ratio > 0:
        for i in range(1, args.n_views):
            inx = np.arange(data_X[i].shape[0])
            rng.shuffle(inx)
            inx = inx[0 : int(args.m_ratio * data_X[i].shape[0])]
            is_align[inx] = 0
            _inx = np.array(inx)
            rng.shuffle(_inx)
            data_X[i][inx] = data_X[i][_inx]
            label_y_view2[inx] = label_y_view2[_inx]
            id2[inx] = id1[_inx]

    noisy_data_X = None
    if args.c_ratio > 0:
        noisy_data_X = [data.copy() for data in data_X]
        N = noisy_data_X[0].shape[0]
        for i in range(args.n_views):
            inx = np.arange(N)
            rng.shuffle(inx)
            inx = inx[0 : int(args.c_ratio * N)]
            std = np.std(noisy_data_X[i], axis=0)
            mean = np.mean(noisy_data_X[i], axis=0)
            noise = np.random.normal(
                loc=mean,
                scale=std,
                size=noisy_data_X[i][inx].shape
            )
            noisy_data_X[i][inx] += noise    

    if args.data_norm == "standard":
        for i in range(args.n_views):
            data_X[i] = skp.scale(data_X[i])
            if noisy_data_X is not None:
                noisy_data_X[i] = skp.scale(noisy_data_X[i])
    elif args.data_norm == "l2-norm":
        for i in range(args.n_views):
            data_X[i] = skp.normalize(data_X[i])
            if noisy_data_X is not None:
                noisy_data_X[i] = skp.normalize(noisy_data_X[i])
    elif args.data_norm == "min-max":
        for i in range(args.n_views):
            data_X[i] = skp.minmax_scale(data_X[i])
            if noisy_data_X is not None:
                noisy_data_X[i] = skp.minmax_scale(noisy_data_X[i])

    return data_X, noisy_data_X, label_y, label_y_view2, id1, id2, is_align, raw_data, raw_label


def load_dataset(args):
    data, noisy_data_X, label1, label2, id1, id2, is_align, raw_data, raw_label = load_mat(args)
    if noisy_data_X is not None:
        train_dataset = MultiviewDataset(args.n_views, noisy_data_X, label1, label2, id1, id2, is_align)
    else:
        train_dataset = MultiviewDataset(args.n_views, data, label1, label2, id1, id2, is_align)
    test_dataset = MultiviewDataset(args.n_views, data, label1, label2, id1, id2, is_align)
    return train_dataset, test_dataset


class MultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, n_views, data_X, label1, label2, id1, id2, is_align):
        super(MultiviewDataset, self).__init__()
        self.n_views = n_views
        self.data = data_X
        self.label1 = label1 - np.min(label1)
        self.label2 = label2 - np.min(label2)
        self.id1 = id1
        self.id2 = id2
        self.is_align = is_align

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.n_views):
            data.append(torch.tensor(self.data[i][idx].astype("float32")))
        label1 = torch.tensor(self.label1[idx], dtype=torch.long)
        label2 = torch.tensor(self.label2[idx], dtype=torch.long)
        id1 = torch.tensor(self.id1[idx], dtype=torch.long)
        id2 = torch.tensor(self.id2[idx], dtype=torch.long)
        label_align = torch.tensor(self.is_align[idx], dtype=torch.long)
        return idx, data, label1, label2, id1, id2, label_align

