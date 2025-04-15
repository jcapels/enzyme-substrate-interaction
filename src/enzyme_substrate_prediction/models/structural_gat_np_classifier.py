import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch import float32, float64, nn
from rdkit import Chem
import numpy as np
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
import torchmetrics

from enzyme_substrate_prediction.models.np_classifier import NPClassifierDNN

def get_3d_coordinates(mol):
    """Generate 3D coordinates for a molecule using RDKit."""
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return coords

def atom_features(atom, idx, conformer):
    """Generate features for a given atom."""
    x, y, z = conformer.GetAtomPosition(idx)
    return np.array([
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetHybridization().real,
        atom.GetImplicitValence(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        int(atom.GetIsAromatic()),
        x,
        y,
        z
    ])

def bond_3d_distance(mol, bond):
    conf = mol.GetConformer()
    u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    pos_u = np.array(conf.GetAtomPosition(u))
    pos_v = np.array(conf.GetAtomPosition(v))
    return np.linalg.norm(pos_u - pos_v)

def bond_features(bond, mol):
    """Generate features for a given bond."""
    bond_types = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2,
                  Chem.rdchem.BondType.TRIPLE: 3, Chem.rdchem.BondType.AROMATIC: 4,
                  Chem.rdchem.BondType.DATIVE: 5}

    distance = bond_3d_distance(mol, bond)

    return np.array([
        bond_types[bond.GetBondType()],
        int(bond.GetIsConjugated()),
        int(bond.IsInRing()),
        distance  # Add the bond distance feature here
    ])

def get_three_d_descriptors(mol):
    from rdkit.Chem.Descriptors3D import CalcMolDescriptors3D
    descriptors = CalcMolDescriptors3D(mol)
    return descriptors

def mol_to_pyg_graph(mol):
    """Converts an RDKit molecule to a PyTorch Geometric Data object."""
    num_atoms = mol.GetNumAtoms()
    edge_index = []
    edge_attr = []
    node_features = []
    conformer = mol.GetConformer()

    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        node_features.append(atom_features(atom, i, conformer))

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((u, v))
        edge_index.append((v, u))
        edge_attr.append(bond_features(bond, mol))
        edge_attr.append(bond_features(bond, mol))

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

class AttentionBlock(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.f_q = nn.Linear(hid_dim, hid_dim)
        self.f_k = nn.Linear(hid_dim, hid_dim)
        self.f_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])) #.cuda()

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        Q = self.f_q(query)
        K = self.f_k(key)
        V = self.f_v(value)
    
        Q = Q.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)
        K_T = K.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3).transpose(2, 3)
        V = V.view(batch_size, self.n_heads, self.hid_dim // self.n_heads).unsqueeze(3)

        self.scale = self.scale.to(V.device)

        energy = torch.matmul(Q, K_T) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        weighter_matrix = torch.matmul(attention, V)

        weighter_matrix = weighter_matrix.permute(0, 2, 1, 3).contiguous()

        weighter_matrix = weighter_matrix.view(batch_size, self.n_heads * (self.hid_dim // self.n_heads))

        weighter_matrix = self.do(self.fc(weighter_matrix))

        return weighter_matrix

class MolecularGAT3D(pl.LightningModule):
    def __init__(self, in_dim, num_heads_gat, num_heads_attention, dropout_gat=0.1, activation_attention = nn.ReLU(), 
                 enzyme_model_path="esm2_3b.pt", use_descriptors=True, prediction_layers=[2560, 1024], dropout_attention=0.1, learning_rate=1e-4, 
                 num_heads_descriptors_cross_attention=1, compounds_model_path="np_classifier.ckpt"):
        super(MolecularGAT3D, self).__init__()
        self.save_hyperparameters()

        hidden_dim = 256
        self.gat1 = GATConv(in_dim, hidden_dim, heads=num_heads_gat, dropout=dropout_gat)
        self.gat2 = GATConv(hidden_dim * num_heads_gat, hidden_dim, heads=1, dropout=dropout_gat)

        self.accuracy = torchmetrics.Accuracy(task="binary")
        self.f1 = torchmetrics.F1Score(task="binary")
        self.recall = torchmetrics.Recall(task="binary")
        self.precision = torchmetrics.Precision(task="binary")
        
        self.ec_number_predictor = nn.Sequential()
        self.ec_number_predictor.add_module("fc_initial", nn.Linear(2560, 2560))
        self.ec_number_predictor.add_module("relu",nn.ReLU())
        self.ec_number_predictor.add_module("batch_norm_initial", nn.BatchNorm1d(2560))
        
        self.ec_number_predictor.load_state_dict(torch.load(enzyme_model_path, map_location=self.device), strict=False)
        self.ec_number_predictor = self.ec_number_predictor.eval()

        assert hidden_dim % num_heads_attention == 0
        self.cross_attention_block = AttentionBlock(2560, num_heads_attention, dropout_attention)
        self.cross_attention_block_descriptors = AttentionBlock(1105, num_heads_descriptors_cross_attention, dropout_attention)

        self.mol_descriptors_batch_norm = nn.BatchNorm1d(849)
        self.protein_descriptors_batch_norm = nn.BatchNorm1d(1105)

        self.compounds_model = NPClassifierDNN()
        self.compounds_model = self.compounds_model.eval()

        np_classifier_model_pretrained_dict = torch.load(compounds_model_path, map_location="cpu")
        np_classifier_model_pretrained_dict = {k.replace("np_classifier_model.", ""): v 
                                               for k, v in np_classifier_model_pretrained_dict["state_dict"].items() 
                                               if k.replace("np_classifier_model.", "") 
                                               in self.compounds_model.state_dict()}
        self.compounds_model.load_state_dict(np_classifier_model_pretrained_dict)

        new_module_2 = nn.Sequential()
        new_module_2.add_module("compounds_head_0", nn.Linear(1536, 2560))
        new_module_2.add_module("compounds_head_0_relu", nn.ReLU())
        new_module_2.add_module("compounds_head_0_batch_norm", nn.BatchNorm1d(2560))
        self.compounds_additional_head = new_module_2

        self.activation = activation_attention
        self.loss_fn = nn.BCELoss()

        if use_descriptors:
            input_dim = 2560 + 1536 + 1105 + 1105
        else:
            input_dim = 2560 + 1536

        self.use_descriptors = use_descriptors

        new_module = nn.Sequential()
        new_module.add_module("prediction_layers_0", nn.Linear(input_dim, prediction_layers[0]))
        new_module.add_module("prediction_layers_0_relu", nn.ReLU())
        # new_module.add_module("prediction_layers_0_batch_norm", nn.BatchNorm1d(prediction_layers[0]))

        if len(prediction_layers)>1:
            for i, layer in enumerate(prediction_layers[1:]):
                new_module.add_module(f"prediction_layers_{i+1}", nn.Linear(prediction_layers[i], layer))
                new_module.add_module(f"prediction_layers_{i+1}_relu", nn.ReLU())
        
            new_module.add_module(f"prediction_layers_{i+1}_batch_norm", nn.BatchNorm1d(layer))

        new_module.add_module(f"prediction", nn.Linear(prediction_layers[-1], 1))
        new_module.add_module(f"prediction_sigmoid", nn.Sigmoid())

        self.prediction_head = new_module
        self.learning_rate = learning_rate
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        h = self.gat1(x, edge_index, edge_attr)
        h = self.activation(h)

        h = self.gat2(h, edge_index, edge_attr)
        h = self.activation(h)

        graph_repr = global_mean_pool(h, batch)

        protein_embedding = data.protein_embedding
        np_classifier_fp = data.fp
        batch_size = protein_embedding.shape[0] // 2560
        protein_embedding = protein_embedding.reshape(batch_size, 2560)
        np_classifier_fp = np_classifier_fp.reshape(batch_size, 6144)
        with torch.no_grad():
            protein_embedding = self.ec_number_predictor(protein_embedding)
            np_prediction, compounds_embedding_raw = self.compounds_model([np_classifier_fp], return_embedding=True)

        compounds_embedding = self.compounds_additional_head(compounds_embedding_raw)

        if self.use_descriptors:
            mol_descriptors = data.mol_descriptors.reshape(batch_size, 849)
            mol_descriptors = self.mol_descriptors_batch_norm(mol_descriptors)
            protein_descriptors = data.protein_descriptors.reshape(batch_size, 1105)
            protein_descriptors = self.protein_descriptors_batch_norm(protein_descriptors)
            mol_descriptors = torch.concat((graph_repr, mol_descriptors), dim=1)
            x_descriptors = self.cross_attention_block_descriptors(protein_descriptors, mol_descriptors, mol_descriptors)

        x = self.cross_attention_block(compounds_embedding, protein_embedding, protein_embedding)
        
        if self.use_descriptors:
            x = torch.concat((x, x_descriptors, protein_descriptors, compounds_embedding_raw), dim=1)

        x = self.prediction_head(x)
        return x

    def training_step(self, batch, batch_idx):
        
        y_pred = self(batch)
        y_pred = y_pred.reshape(y_pred.shape[0],)
        # y_pred = torch.nan_to_num(y_pred, nan=0.0)

        loss = self.loss_fn(y_pred, batch.y)

        self.log("train_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        
        accuracy = self.accuracy(y_pred, batch.y)
        f1 = self.f1(y_pred, batch.y)
        recall = self.recall(y_pred, batch.y)
        precision = self.precision(y_pred, batch.y)

        self.log("train_accuracy", accuracy, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        self.log("train_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        self.log("train_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        
        self.log("train_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])

        return loss
    
    def predict_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_pred = y_pred.reshape(y_pred.shape[0],)
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        return y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_pred = y_pred.reshape(y_pred.shape[0],)
        y_pred = torch.nan_to_num(y_pred, nan=0.0)
        loss = self.loss_fn(y_pred, batch.y)
        self.log('test_loss', loss, batch_size=y_pred.shape[0])
        self.accuracy(y_pred, batch.y)
        self.log('test_acc', self.accuracy, batch_size=y_pred.shape[0])
        return loss
    
    
    
    def validation_step(self, batch, batch_idx):
        y_pred = self(batch)
        y_pred = y_pred.reshape(y_pred.shape[0],)
        y_pred = torch.nan_to_num(y_pred, nan=0.0)

        loss = self.loss_fn(y_pred, batch.y)

        self.log("val_loss", loss.item(), on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        
        accuracy = self.accuracy(y_pred, batch.y)
        f1 = self.f1(y_pred, batch.y)
        recall = self.recall(y_pred, batch.y)
        precision = self.precision(y_pred, batch.y)

        self.log("val_accuracy", accuracy, on_epoch=True, 
                 prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        self.log("val_f1", f1, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        self.log("val_recall", recall, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        
        self.log("val_precision", precision, on_epoch=True,
                    prog_bar=True, logger=True, sync_dist=True, batch_size=y_pred.shape[0])
        
        return loss
    
from sklearn.preprocessing import StandardScaler

# Example dataset class
class MolecularGraphDataset(Dataset):
    def __init__(self, mol_dict, mol_descriptors, protein_descriptors, proteins_embeddings_dict, np_classifier_fp_dict, interactions, labels, mol_scaler=None, protein_scaler=None):
        self.graphs = []
        self.mol_dict = mol_dict
        self.proteins_embeddings_dict = proteins_embeddings_dict
        self.protein_descriptors = protein_descriptors
        self.np_classifier_fp_dict = np_classifier_fp_dict
        self.interactions = interactions
        self.labels = labels
        self.mol_descriptors=mol_descriptors


        self.unique_protein_ids = set()
        self.unique_compound_ids = set()
        # Collect unique protein and compound IDs
        for protein_id, compound_id in self.interactions:
            self.unique_protein_ids.add(protein_id)
            self.unique_compound_ids.add(compound_id)

        if mol_scaler and protein_scaler:
            self.mol_scaler = mol_scaler
            self.protein_scaler = protein_scaler
        else:
            self.scale_fit()

        self.scale_transform()

        for key, mol in self.mol_dict.items():
            # try:
            data = mol_to_pyg_graph(mol)
            self.mol_dict[key] = data
            # except Exception as e:
            #     print(e.with_traceback())
            #     pass

    def scale_fit(self):
        # Initialize the scalers
        self.mol_scaler = StandardScaler()
        self.protein_scaler = StandardScaler()

        # Extract unique descriptors
        unique_protein_descriptors = [self.protein_descriptors[protein_id] for protein_id in self.unique_protein_ids]
        unique_compound_descriptors = [self.mol_descriptors[compound_id] for compound_id in self.unique_compound_ids]

        # Fit the scalers on the unique descriptors
        self.protein_scaler.fit(unique_protein_descriptors)
        self.mol_scaler.fit(unique_compound_descriptors)

    def scale_transform(self):

        # Transform the original descriptors using the fitted scalers
        self.protein_descriptors = {protein_id: self.protein_scaler.transform([self.protein_descriptors[protein_id]])[0]
                                    for protein_id in self.unique_protein_ids}
        self.mol_descriptors = {compound_id: self.mol_scaler.transform([self.mol_descriptors[compound_id]])[0]
                                for compound_id in self.unique_compound_ids}

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        protein_id, compound_id = self.interactions[idx]
        label = self.labels[idx]
        graph = self.mol_dict[compound_id]
        graph.y = torch.tensor([label], dtype=torch.float32)
        graph.protein_embedding = torch.tensor(self.proteins_embeddings_dict[protein_id], dtype=torch.float32)
        graph.protein_descriptors = torch.tensor(self.protein_descriptors[protein_id], dtype=torch.float32)
        graph.fp = torch.tensor(self.np_classifier_fp_dict[compound_id], dtype=torch.float32)
        graph.idx = idx
        graph.mol_descriptors = torch.tensor(self.mol_descriptors[compound_id], dtype=torch.float32)

        return graph

def collate_fn(batch):
    return Batch.from_data_list(batch)


