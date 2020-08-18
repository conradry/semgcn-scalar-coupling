from __future__ import division
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

#define the device globally for all operations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#from the presentation it appears that the model uses a standard resnet Bottleneck block
#without ever downsampling, so we can exclude that option in this case
    
class GraphConv(nn.Module):
    def __init__(self, nin, nout, dilation=1):
        super(GraphConv, self).__init__()
        self.conv = nn.Conv1d(nin, nout, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(nout))
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(True)
        self.dilation = dilation
        
    def forward(self, inps):
        #the operation is to first mask the elements in
        #x based on the number of bonds
        #this information is found in A, the value in A corresponds
        #to the shortest number of hops between atom0 and atom1
        #we can apply a dilation operation by considering the 
        #neighborhood such that n_hops = dilation rate
        
        A = inps['graph']
        x = inps['features']
        
        #now we do elementwise multiplication to
        #zero out elements not in an atoms neighborhood
        #hopefully, broadcasting semantics are correct here
        y = torch.zeros(x.size(), dtype=torch.float32).cuda()
        x = torch.where((A == self.dilation) | (A == -1), x, y)
            
        #we we apply a 1x1 convolution to each element in the tensor
        x = self.conv(x)
        outsize = x.size()
        
        #summing over the third axis is the same as a sum over the atoms
        #receptive field
        x = x.sum(dim=-1)
        x = x + self.bias.view(1, -1).expand_as(x)
        
        #now, we expand back to the original size
        #each row will have the same data
        #x = x.unsqueeze(-2).expand(outsize)
        
        #y = torch.zeros(x.size(), dtype=torch.float32).cuda()
        #x = torch.where((A == self.dilation) | (A == -1), x, y)
        #print(x[0, 0])
  
        #the last step here is to expand the tensor back to it's original shape
        return {'graph': A, 'features': self.relu(self.bn(x))}
    
class DistConv(nn.Module):
    def __init__(self, nin, nout, dilation=1):
        super(DistConv, self).__init__()
        self.conv = nn.Conv1d(nin, nout, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(nout))
        self.bn = nn.BatchNorm1d(nout)
        self.relu = nn.ReLU(True)
        self.dilation = dilation
        
    def forward(self, inps):
        #the operation is to first mask the elements in
        #x based on the number of bonds
        #this information is found in A, the value in A corresponds
        #to the shortest number of hops between atom0 and atom1
        #we can apply a dilation operation by considering the 
        #neighborhood such that n_hops = dilation rate
        
        A = inps['graph']
        x = inps['features']
        
        #now we do elementwise multiplication to
        #zero out elements not in an atoms neighborhood
        #hopefully, broadcasting semantics are correct here
        y = torch.zeros(x.size(), dtype=torch.float32).cuda()
        x = torch.where((A > self.dilation) | (A == -1), x, y)
            
        #we we apply a 1x1 convolution to each element in the tensor
        x = self.conv(x)
        outsize = x.size()
        
        #summing over the third axis is the same as a sum over the atoms
        #receptive field
        x = x.sum(dim=-1)
        x = x + self.bias.view(1, -1).expand_as(x)
        
        #now, we expand back to the original size
        #each row will have the same data
        #x = x.unsqueeze(-2).expand(outsize)
        
        #y = torch.zeros(x.size(), dtype=torch.float32).cuda()
        #x = torch.where((A == self.dilation) | (A == -1), x, y)
        #print(x[0, 0])
  
        #the last step here is to expand the tensor back to it's original shape
        return {'graph': A, 'features': self.relu(self.bn(x))}
    
class GraphBottleneck(nn.Module):

    def __init__(self, nin, compress=2, dilation=1):
        super(GraphBottleneck, self).__init__()
        #the basic resnet-style block that they use
        #starts with batchnorm, elu, project down
        self.dilation = dilation
        self.relu = nn.ELU(inplace=True)
        
        self.bn1 = nn.BatchNorm2d(nin)
        self.project_down = nn.Conv2d(nin, nin // compress, 1)
        
        #next we have the relu, bn, graph conv
        self.bn2 = nn.BatchNorm2d(nin // compress)
        self.graph_conv = GraphConv(nin // compress, nin // compress, dilation=dilation)
        
        #last we have the relu, bn, project up
        self.bn3 = nn.BatchNorm2d(nin // compress)
        self.project_up = nn.Conv2d(nin // compress, nin, 1)
 
    def forward(self, inps):
        
        A = inps['graph']
        x = inps['features']
        
        #first mask the input to match receptive field
        y = torch.zeros(x.size(), dtype=torch.float32).cuda()
        x = torch.where(A < self.dilation, x, y)
        
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.project_down(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        inps = {'graph': A, 'features': out}
        out = self.graph_conv(inps)
        out = out['features']
        
        out = self.bn3(out)
        out = self.relu(out)
        out = self.project_up(out)
        
        return {'graph': A, 'features': out + identity}
    
class CouplingConv(nn.Module):
    def __init__(self, nin, nout):
        super(CouplingConv, self).__init__()
        self.conv = nn.Conv2d(nin, nout, 1)
    
    def forward(self, inps):
        #to get a symmetric output matrix, we take
        #a column and it's transpose, expand them into different shapes
        #and then sum
        #print(x[0, 0])
        #A = inps['graph']
        x = inps['features']
        #print(x[0, 0])
        
        #y = torch.zeros(x.size(), dtype=torch.float32).cuda()
        #print(x[0, 0])
        #x = torch.where(A != 0, x, y)
        #print(x[0, 0])
        #print(A[0, 0] != 0)
        #print(x[0, 0])
        
        #insize = x.size()
        #x1 = x[:, :, 0]
        #print(x1[0, 0])
        
        #xi = x1.unsqueeze(-1).expand(insize)
        #xj = x1.unsqueeze(-2).expand(insize)
        #print(xi[0, 0], xj[0, 0])
        
        #sym = xi + xj
        #print(sym[0, 0])
        #sym = self.conv(sym)
        
        sym = self.conv(x)
        return sym
        
    
class Model(nn.Module):
    def __init__(self, bond_embedding_size, type_embedding_size, inplanes=32, n_blocks=18):
        super(Model, self).__init__()
        
        #let's define two embedding layers, 1 for our bonds and the other for coupling types
        #15 types of bonds, 8 types of couplings, always include +1 to account for zeros
        self.bond_embed = nn.Embedding(15 + 1, bond_embedding_size, padding_idx=0)
        self.type_embed = nn.Embedding(8 + 1, type_embedding_size, padding_idx=0)
        self.dilations = [1, 2, 3]
        
        self.inplanes = inplanes
        in_features = bond_embedding_size + 1 # + type_embedding_size
        #self.network = nn.Sequential(
        #    nn.Conv2d(in_features, self.inplanes, 1),
        #    nn.BatchNorm2d(self.inplanes),
        #    nn.ReLU(True),
        #    nn.Conv2d(self.inplanes, self.inplanes, 1),
        #    nn.BatchNorm2d(self.inplanes),
        #    nn.ReLU(True),
        #    nn.Conv2d(self.inplanes, self.inplanes, 1),
        #    nn.BatchNorm2d(self.inplanes),
        #    nn.ReLU(True),
        #    nn.Conv2d(self.inplanes, self.inplanes, 1),
        #    nn.BatchNorm2d(self.inplanes),
        #    nn.ReLU(True),
        #    nn.Conv2d(self.inplanes, self.inplanes, 1),
        #    nn.BatchNorm2d(self.inplanes),
        #    nn.ReLU(True),
        #    nn.Conv2d(self.inplanes, 1, 1)
        #)
        #typical for resnet to put a large receptive field convolution
        #at the beginning, may be unnecessary
        
        #self.network = nn.Sequential(
        #    GraphConv(in_features, self.inplanes, 3),
        #    CouplingConv(self.inplanes, 1)
        #)
        
        #self.elu = nn.ELU()
        #self.conv_in1 = GraphConv(in_features, self.inplanes, dilation=1)
        #self.bn1 = nn.BatchNorm2d(self.inplanes)
        #self.conv_in2 = GraphConv(in_features, self.inplanes, dilation=2)
        #self.bn2 = nn.BatchNorm2d(self.inplanes)
        #self.conv_in3 = GraphConv(in_features, self.inplanes, dilation=3)
        #self.bn3 = nn.BatchNorm2d(self.inplanes)
        #self.conv_down = nn.Conv2d(self.inplanes * 3, self.inplanes, 1)
        
        #block = GraphBottleneck
        #self.network = self._make_layer(block, n_blocks)
        
        #self.coupling = nn.Sequential(
        #    nn.BatchNorm2d(self.inplanes),
        #    nn.ELU(True),
        #    CouplingConv(self.inplanes, self.inplanes)
        #)
        
        #self.final = nn.Sequential(
        #    nn.BatchNorm2d(self.inplanes + type_embedding_size),
        #    nn.ELU(True),
        #    nn.Conv2d(self.inplanes + type_embedding_size, 1, 1)
        #)
        
        self.conv1 = GraphConv(in_features, self.inplanes, 1)
        self.conv2 = GraphConv(in_features, self.inplanes, 2)
        self.conv3 = GraphConv(in_features, self.inplanes, 3)
        self.conv4 = GraphConv(in_features, self.inplanes, 4)
        self.conv5 = GraphConv(in_features, self.inplanes, 5)
        self.conv6 = GraphConv(in_features, self.inplanes, 6)
        self.conv7 = GraphConv(in_features, self.inplanes, 7)
        self.conv8 = GraphConv(in_features, self.inplanes, 8)
        self.conv9 = DistConv(in_features, self.inplanes, 8)
        
        self.head = nn.Sequential(
            nn.Linear(self.inplanes * 9 + 5, 256),
            nn.ReLU(True),
            nn.Dropout(0.9),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.9),
            nn.Linear(128, 1)
        )
        
    def _make_layer(self, block, n_blocks):
        layers = []
        for ix in range(n_blocks):
            dilation = self.dilations[ix % len(self.dilations)]
            layers.append(block(self.inplanes, compress=2, dilation=dilation))
            
        return nn.Sequential(*layers)
    
    def forward(self, distance0, distance1, bond0, bond1, ctype, graph0, graph1):
        #embeddings puts the channel dimension last, let's change that
        bsz = distance0.size(0)
        bond0 = self.bond_embed(bond0).transpose(-1, 1)
        bond1 = self.bond_embed(bond1).transpose(-1, 1)
        ctype = self.type_embed(ctype).transpose(-1, 1)
        
        features0 = torch.cat([distance0, bond0], dim=1)
        inps0 = {'graph': graph0, 'features': features0}
        
        features1 = torch.cat([distance1, bond1], dim=1)
        inps1 = {'graph': graph1, 'features': features1}
        
        ns = []
        ns.append(self.conv1(inps0)['features'] + self.conv1(inps1)['features'])
        ns.append(self.conv2(inps0)['features'] + self.conv2(inps1)['features'])
        ns.append(self.conv3(inps0)['features'] + self.conv3(inps1)['features'])
        ns.append(self.conv4(inps0)['features'] + self.conv4(inps1)['features'])
        ns.append(self.conv5(inps0)['features'] + self.conv5(inps1)['features'])
        ns.append(self.conv6(inps0)['features'] + self.conv6(inps1)['features'])
        ns.append(self.conv7(inps0)['features'] + self.conv7(inps1)['features'])
        ns.append(self.conv8(inps0)['features'] + self.conv8(inps1)['features'])
        ns.append(self.conv9(inps0)['features'] + self.conv9(inps1)['features'])
        ns.append(ctype)
        
        ns = torch.cat(ns, dim=1)
        ns = ns.view(bsz, -1)
        
        out = self.head(ns)
        
        #features = torch.cat([distance, bond, ctype], dim=1)
        #out = self.network(distance)
        #to enable use of sequential, wrap inputs in a dict
        
        #out = self.network(inps)
        
        #in1 = self.conv_in1(inps)['features']
        #in1 = self.elu(self.bn1(in1))
        #in2 = self.conv_in2(inps)['features']
        #in2 = self.elu(self.bn2(in2))
        #in3 = self.conv_in3(inps)['features']
        #in3 = self.elu(self.bn3(in3))
        
        #out = self.conv_down(torch.cat([in1, in2, in3], dim=1))
        #out = {'graph': graph, 'features': out}
        
        #out = self.network(inps)
        #out = out['features']
        
        #out = self.coupling(out)
        
        #out = torch.cat([out, ctype], dim=1)
        #out = self.final(out)
        #print(out[0, 0])
        
        #return the symetric matrix prediction
        return out
    
class SmoothL1(nn.Module):
    def __init__(self, beta=0.11):
        super(SmoothL1, self).__init__()
        """At later time check if we should handle error by image or batch"""
        self.beta = beta
        
    def forward(self, output, target, size_average=True):
        #first we need to filter our all the zeros from padding and 
        #empty scalar constants
        
        output = output.view(-1)
        target = target.view(-1)
        
        
        n = torch.abs(output - target)
        cond = n < self.beta
        loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        
        if size_average:
            return loss.mean()
        
        return loss.sum()
    
class SmoothL1Types(nn.Module):
    def __init__(self, beta=0.11):
        super(SmoothL1Types, self).__init__()
        """At later time check if we should handle error by image or batch"""
        self.beta = beta
        
    def forward(self, output, target, types, size_average=True):
        #first we need to filter our all the zeros from padding and 
        #empty scalar constants
        
        output = output.view(-1)
        types = types.view(-1)
        target = target.view(-1)
        
        output = output[target != 0]
        target = target[target != 0]
        types = types[types != 0]
        
        type_loss = torch.tensor([0]).cuda().float()
        for i in range(1, 9):
            type_output = output[types == i]
            type_target = target[types == i]
            n = torch.abs(output - target)
            cond = n < self.beta
            loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
            type_loss = type_loss + loss.mean()  
        
        return type_loss
    
class Learner:
    def __init__(self, model, loss, optimizer, train, valid=None):
        self.model = model.to(device)
        self.loss = loss
        self.optimizer = optimizer
        self.train = train
        self.valid = valid
        
        #only metric besides loss that we care about is MAE
        #we'll store the values in a list that get's refreshed
        #when metrics are printed
        self.train_metrics = {'loss': [], 'mae': []}
        self.valid_metrics = {'val loss': [], 'val mae': []}
        
    def load_batch(self, dataloader):
        return iter(dataloader).next()
    
    def extract_data(self, batch):
        #4 fields that we care about in the batch: distance, bond, type, and target
        distance0 = batch['distance0'].unsqueeze(1).to(device)
        distance1 = batch['distance0'].unsqueeze(1).to(device)
        bond0 = batch['bond0'].to(device)
        bond1 = batch['bond1'].to(device)
        ctype = batch['type'].to(device)
        graph0 = batch['graph0'].unsqueeze(1).to(device)
        graph1 = batch['graph1'].unsqueeze(1).to(device)
        
        #distances and targets need to have a channel dimension added, bonds and types
        #have channel dimension added when then pass through the embedding layer
        
        return distance0, distance1, bond0, bond1, ctype, graph0, graph1
    
    def extract_target(self, batch):
        return batch['target'].unsqueeze(1).to(device)
    
    def batch_train(self):
        #first get a batch of data from training set
        batch = self.load_batch(self.train)
        in_data = self.extract_data(batch)
        target = self.extract_target(batch)
        
        #now predict and update the model
        self.optimizer.zero_grad()
        output = self.model(*in_data)
        l = self.loss(output, target, in_data[4])
        l.backward()
        
        self.optimizer.step()
        
        #lastly calculate the mae
        types = batch['type']
        mae = calculate_mae(output.detach(), target.detach(), types)
        return l.item(), mae
    
    def batch_eval(self):
        #first get a batch of data from training set
        with torch.no_grad():
            batch = self.load_batch(self.valid)
            in_data = self.extract_data(batch)
            target = self.extract_target(batch)

            #now predict and update the model
            output = self.model.eval()(*in_data)
            l = self.loss(output, target, in_data[4])

            #lastly calculate the mae
            types = batch['type']
            mae = calculate_mae(output.detach(), target.detach(), types)
            
        return l.item(), mae
            
    def batch_predict(self, batch):
        with torch.no_grad():
            in_data = self.extract_data(batch)
            
            #now predict and update the model
            output = self.model.eval()(*in_data)
            
        return output.detach().cpu(), target.detach().cpu()
    
    def print_metrics(self, metric_dict):
        for m in metric_dict.keys():
            print('{}: {}'.format(m, metric_dict[m][-1]))
    
    def training(self, epochs, iters=None):
        #if no fixed number of iterations is specified we just use the length
        #of the training dataset
        if not iters:
            iters = len(self.train)
        
        rl = rmae = 0
        for e in range(epochs):
            rl = rmae = 0
            for ix in tqdm(range(iters)):
                loss, mae = self.batch_train()
                rl += loss
                rmae += mae
                
            self.train_metrics['loss'].append(rl / iters)
            self.train_metrics['mae'].append(rmae / iters)
            self.print_metrics(self.train_metrics)
            
            if self.valid:
                self.evaluate()      
                
    def evaluate(self):
        assert (self.valid is not None), "No validation data to evaluate!"
        
        rl = rmae = 0
        for ix in range(len(self.valid)):
            val_loss, val_mae = self.batch_eval()
            rl += val_loss
            rmae += val_mae

        self.valid_metrics['val loss'].append(rl / len(self.valid))
        self.valid_metrics['val mae'].append(rmae / len(self.valid))
        self.print_metrics(self.valid_metrics)
        
    def batchify(self, tensor_list):
        """Add a batch dimension to each tensor in the list"""
        return [tensor.unsqueeze(0) for tensor in tensor_list]
        
    def create_submission(self, test_data, test_df):
        ids = []
        preds = []
        for data_dict in tqdm(test_data):
            molecule = data_dict['fname'].split('.')[0] #remove .npy suffix
            distances = data_dict['distance'].to(device).unsqueeze(0)
            bonds = data_dict['bond'].to(device)
            types = data_dict['type'].to(device)
            
            #now we batchify the data and feed it to the model
            batched = self.batchify([distances, bonds, types])
            with torch.no_grad():
                output = self.model.eval()(*batched)
                output = output.squeeze(0).detach().cpu().numpy()
            
            #next we need lookup the filename, atom indices, and interaction id
            #from the test dataframe
            molecule_df = test_df[test_df.molecule_name == molecule]
            atom_pairs = list(zip(molecule_df.atom_index_0, molecule_df.atom_index_1))
            
            ids.append(molecule_df.id)
            preds.append([output[i][j] for i, j in atom_pairs])
            
        ids = np.concatenate(ids)
        preds = np.concatenate(preds)
        
        pred_df = pd.DataFrame(data=zip(ids, preds), columns=['id', 'scalar_coupling_constant'])
        return pred_df.sort_values(by='id')
        
def calculate_mae(output, target, types):
    #first, let's flatten our input
    output = output.view(-1)
    target = target.view(-1)
    types = types.view(-1)
    
    #we know that there are 8 types
    #we cycle through each one to get mae for that type
    type_aes = []
    for i in range(1, 9):
        type_mask = types == i
        type_output = output[type_mask]
        if len(type_output) != 0:
            type_target = target[type_mask]
            ae = torch.log(torch.mean(torch.abs(type_output - type_target)))
            type_aes.append(ae)
    
    type_aes = torch.tensor(type_aes)
    return torch.mean(type_aes)
                