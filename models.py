import math
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sklearn.cluster import AgglomerativeClustering



#######################################
class ResChunk(nn.Module):
    def __init__(self, args):
        super(ResChunk, self).__init__()

        self.joint_grouping = JointGrouping(args)

        self.num_edge_types = args.num_edge_types
        self.gumbel_temp = args.gumbel_temp
        self.normalize_kl = args.normalize_kl
        self.kl_coef = args.kl_coef
        
        prior = np.zeros(self.num_edge_types)
        prior.fill((1 - args.no_edge_prior)/(self.num_edge_types - 1))
        prior[0] = args.no_edge_prior
        log_prior = torch.FloatTensor(np.log(prior))
        log_prior = torch.unsqueeze(log_prior, 0)
        self.log_prior = torch.unsqueeze(log_prior, 0)

        p_dropout = args.mscale_dropout
        input_feature = args.input_seq_len // 2 if args.even else args.input_seq_len
        output_feature = args.target_seq_len // 2 if args.even else args.target_seq_len
        n_chunks = 6       # number of sequence chunks which is fixed; if changes, residual blocks must change 
        self.chs = output_feature // n_chunks
        self.ch1 = output_feature - self.chs*(n_chunks-1)
        if args.data_type == 'aa':
            self.dim = 3
        else:
            raise NotImplementedError

        hid_size = args.gcn_hidden
        final_out_noden = args.num_joints * self.dim

        self.start_gcn = PreGCN(input_feature=input_feature, hidden_feature=hid_size, node_n=final_out_noden, p_dropout=p_dropout)
        
        self.gc_block_1 = nn.Sequential(
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),  
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),)

        self.gc_block_2 = nn.Sequential(
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),)

        self.gc_block_3 = nn.Sequential(
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),)

        self.gc_block_4 = nn.Sequential(
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),)

        self.gc_block_5 = nn.Sequential(
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),)

        self.gc_block_6 = nn.Sequential(
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),)

        self.gc_block_7 = nn.Sequential(
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),  
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),)

        self.end_gcn_1 = nn.Sequential(
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),  
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            GC_Block(in_features=hid_size, p_dropout=p_dropout, node_n=final_out_noden),
            PostGCN(input_feature=hid_size, hidden_feature=output_feature, node_n=final_out_noden))
        self.end_gcn_2 = PostGCN(input_feature=hid_size, hidden_feature=output_feature, node_n=final_out_noden)
        
        self.agglomer = AgglomerativeClustering().set_params(affinity='l2', linkage='average')
        self.num_joints = args.num_joints
        self.input_feature = input_feature
        self.output_feature = output_feature
        

    def forward(self, x, y=None):
        B = x.shape[0]
        device = x.device
        nj = self.num_joints

        losses = 0.0
        preds = []
        x_scale = x.clone()
        ch1 = self.ch1
        chs = self.chs
        if self.training:
            y_scale = y.clone()
        hard_sample = not self.training

        # scale extraction 
        logits = self.joint_grouping(x.permute(0, 2, 1).view(B, x.shape[2], nj, self.dim))
        edges = gumbel_softmax(logits.reshape(-1, self.num_edge_types), 
                            tau=self.gumbel_temp, hard=hard_sample).view(logits.shape)
        
        edges_c = torch.sigmoid(edges[:, :, 0])
        joint_groups = edges_c.view(B, nj, nj-1)
        for i in range(B):
            squared_rels = torch.zeros(nj, nj).to(device)
            squared_rels[:, :-1] += torch.tril(joint_groups[i], -1)
            squared_rels[:, 1:] += torch.triu(joint_groups[i], 0)
            idxp = self.agglomer.fit_predict(squared_rels.detach().cpu().numpy())
            idx = torch.tensor(idxp)
            for j in range(idxp.max()+1):
                id1 = torch.where(idx == j)[0].long().to(device)
                x_scale.permute(0, 2, 1).view(B, x.shape[2], nj, self.dim)[i, :, id1, 0] = x.permute(0, 2, 1).view(B, x.shape[2], nj, self.dim)[i, :, id1, 0].mean(dim=1).unsqueeze(1).repeat(1, id1.shape[0])
                x_scale.permute(0, 2, 1).view(B, x.shape[2], nj, self.dim)[i, :, id1, 1] = x.permute(0, 2, 1).view(B, x.shape[2], nj, self.dim)[i, :, id1, 1].mean(dim=1).unsqueeze(1).repeat(1, id1.shape[0])
                x_scale.permute(0, 2, 1).view(B, x.shape[2], nj, self.dim)[i, :, id1, 2] = x.permute(0, 2, 1).view(B, x.shape[2], nj, self.dim)[i, :, id1, 2].mean(dim=1).unsqueeze(1).repeat(1, id1.shape[0])
                if self.training:
                    y_scale.permute(0, 2, 1).view(B, y.shape[2], nj, self.dim)[i, :, id1, 0] = y.permute(0, 2, 1).view(B, y.shape[2], nj, self.dim)[i, :, id1, 0].mean(dim=1).unsqueeze(1).repeat(1, id1.shape[0])
                    y_scale.permute(0, 2, 1).view(B, y.shape[2], nj, self.dim)[i, :, id1, 1] = y.permute(0, 2, 1).view(B, y.shape[2], nj, self.dim)[i, :, id1, 1].mean(dim=1).unsqueeze(1).repeat(1, id1.shape[0])
                    y_scale.permute(0, 2, 1).view(B, y.shape[2], nj, self.dim)[i, :, id1, 2] = y.permute(0, 2, 1).view(B, y.shape[2], nj, self.dim)[i, :, id1, 2].mean(dim=1).unsqueeze(1).repeat(1, id1.shape[0])
        
        prob = F.softmax(logits, dim=-1)
        losses += self.kl_categorical(prob).mean()

        x_1 = x.clone() 
        x_2 = x_scale.clone()
    
        pre = self.start_gcn(x_1)  
        
        out1 = torch.cat((self.gc_block_1(pre), pre), dim=-2)  
        out1 = self.chunkponosig(out1) 
        out = self.end_gcn_1(out1)
        y0 = x_1[:, :, -ch1:] + out[:, :, :ch1]
        y0[..., :1] = x_1[:, :, -1:] + out[:, :, :1]
        for h in range(1,ch1):
            y0[..., h:h+1] = y0[:, :, h-1:h] + out[:, :, h:h+1] 
        preds.append(y0)

        out2 = torch.cat((self.gc_block_2(out1), out1), dim=-2)  
        out2 = self.chunkponosig(out2)
        out = self.end_gcn_1(out2)
        y1 = y0[:, :, ch1-chs:] + out[:, :, ch1:ch1+chs] 
        y1[..., :1] = y0[:, :, -1:] + out[:, :, ch1:ch1+1] 
        for h in range(1,chs):
            y1[..., h:h+1] = y1[:, :, h-1:h] + out[:, :, ch1+h:ch1+h+1] 
        preds.append(y1)

        out3 = torch.cat((self.gc_block_3(out2), out2), dim=-2)  
        out3 = self.chunkponosig(out3)
        out = self.end_gcn_1(out3)
        y2 =  y1 + out[:, :, ch1+chs:ch1+(2*chs)]
        y2[..., :1] = y1[:, :, -1:] + out[:, :, ch1+chs:ch1+chs+1]
        for h in range(1,chs):
            y2[..., h:h+1] = y2[:, :, h-1:h] + out[:, :, ch1+chs+h:ch1+chs+h+1]  
        preds.append(y2)

        out4 = torch.cat((self.gc_block_4(out3), out3), dim=-2)
        out4 = self.chunkponosig(out4)
        out = self.end_gcn_1(out4)
        y3 =  y2 + out[:, :, ch1+(2*chs):ch1+(3*chs)]
        y3[..., :1] = y2[:, :, -1:] + out[:, :, ch1+(2*chs):ch1+(2*chs)+1]
        for h in range(1,chs):
            y3[..., h:h+1] = y3[:, :, h-1:h] + out[:, :, ch1+(2*chs)+h:ch1+(2*chs)+h+1]  
        preds.append(y3)

        out5 = torch.cat((self.gc_block_5(out4), out4), dim=-2)  
        out5 = self.chunkponosig(out5)
        out = self.end_gcn_1(out5)
        y4 =  y3 + out[:, :, ch1+(3*chs):ch1+(4*chs)]
        y4[..., :1] = y3[:, :, -1:] + out[:, :, ch1+(3*chs):ch1+(3*chs)+1]
        for h in range(1,chs):
            y4[..., h:h+1] = y4[:, :, h-1:h] + out[:, :, ch1+(3*chs)+h:ch1+(3*chs)+h+1] 
        preds.append(y4)
        
        out6 = torch.cat((self.gc_block_6(out5), out5), dim=-2) 
        out6 = self.chunkponosig(out6)
        out = self.end_gcn_1(out6)
        y5 =  y4 + out[:, :, ch1+(4*chs):]
        y5[..., :1] = y4[:, :, -1:] + out[:, :, ch1+(4*chs):ch1+(4*chs)+1] 
        for h in range(1,chs):
            y5[..., h:h+1] = y5[:, :, h-1:h] + out[:, :, ch1+(4*chs)+h:ch1+(4*chs)+h+1]
        preds.append(y5)

        pred_1 = torch.cat(preds, dim=2)

        out7 = torch.cat((self.gc_block_7(out5), out5), dim=-2)  
        out7 = self.chunkponosig(out7)
        
        pred_2 = self.end_gcn_2(out7) + x_2

        outputs = {"s_1": pred_1, "s_2": pred_2}

        if self.training:
            losses += self.nll_gaussian(y.contiguous(), outputs['s_1'].contiguous())
            losses += self.nll_gaussian(y_scale.contiguous(), outputs['s_2'].contiguous())

        else:
            outputs = outputs['s_1']
            outputs = outputs.permute(0, 2, 1)
            
        return losses, outputs


    def chunkponosig(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        a, _, __ = pono(a)
        y = a * torch.sigmoid(b)
        return y

    def nll_gaussian(self, preds, target, add_const=False):
        prior_variance = 5e-5
        neg_log_p = ((preds - target) ** 2 / (2 * prior_variance))
        if add_const:
            const = 0.5 * np.log(2 * np.pi * prior_variance)
            neg_log_p += const
        return neg_log_p.view(target.size(0), -1).sum() / (target.size(1))

    def kl_categorical(self, preds, eps=1e-16):
        kl_div = preds*(torch.log(preds+eps) - self.log_prior.to(preds.device))
        if self.normalize_kl:     
            return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)
        elif self.normalize_kl_per_var:
            return kl_div.sum() / (self.num_joints * preds.size(0))
        else:
            return kl_div.view(preds.size(0), -1).sum(dim=1)


#######################################
class JointGrouping(nn.Module):
    def __init__(self, args):
        super(JointGrouping, self).__init__()
        self.num_joints = args.num_joints
        if args.data_type == 'aa':
            joint_dim = 3
        else:
            raise NotImplementedError
        self.input_time_steps = args.input_seq_len // 2 if args.even else args.input_seq_len
        inp_size = joint_dim * self.input_time_steps
        hidden_size = args.encoder_hidden
        num_edges = args.num_edge_types
        self.graph_type = args.graph_type
        dropout = args.encoder_dropout
        self.mlp1 = FC_ELU(inp_size, hidden_size, hidden_size, dropout, no_bn=False)
        self.mlp2 = FC_ELU(hidden_size * 2, hidden_size, hidden_size, dropout, no_bn=False)
        self.mlp3 = FC_ELU(hidden_size, hidden_size, hidden_size, dropout, no_bn=False)
        self.mlp4 = FC_ELU(hidden_size * 3, hidden_size, hidden_size, dropout, no_bn=False)
        num_layers = args.encoder_mlp_num_layers
        
        tmp_hidden_size = args.encoder_mlp_hidden
        layers = [nn.Linear(hidden_size, tmp_hidden_size), nn.ELU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(tmp_hidden_size, tmp_hidden_size))
            layers.append(nn.ELU(inplace=True))
        layers.append(nn.Linear(tmp_hidden_size, num_edges))
        self.fc_out = nn.Sequential(*layers)

        self.init_weights()

        edges = np.ones(self.num_joints) - np.eye(self.num_joints)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]
        self.edge2node_mat = nn.Parameter(torch.FloatTensor(encode_onehot(self.recv_edges).transpose()), requires_grad=False)
    
    def node2edge(self, node_embeddings):
        send_embed = node_embeddings[:, self.send_edges, :]
        recv_embed = node_embeddings[:, self.recv_edges, :]
        return torch.cat([send_embed, recv_embed], dim=2)

    def edge2node(self, edge_embeddings):
        incoming = torch.matmul(self.edge2node_mat, edge_embeddings)
        return incoming/(self.num_joints-1)    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)


    def forward(self, inputs):
        x = inputs.transpose(1, 2).contiguous().view(inputs.size(0), inputs.size(2), -1)
        x = self.mlp1(x) 

        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x

        x = self.edge2node(x)
        x = self.mlp3(x)
        x = self.node2edge(x)
        x = torch.cat((x, x_skip), dim=-1) 
        x = self.mlp4(x)
        
        result =  self.fc_out(x)
        return result


#######################################
class PreGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, node_n, p_dropout):
        super(PreGCN, self).__init__()

        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        self.node_n = node_n

        self.gcn = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1d = nn.BatchNorm1d(node_n * hidden_feature)
        self.act_f = nn.Tanh()

        self.do = nn.Dropout(p_dropout)

    def forward(self, x):
        y = self.gcn(x)
        b, n, f = y.shape
        y = self.bn1d(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)
        return y


#######################################
class PostGCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, node_n):
        super(PostGCN, self).__init__()

        self.input_feature = input_feature
        self.hidden_feature = hidden_feature
        self.node_n = node_n

        self.gcn = GraphConvolution(input_feature, hidden_feature, node_n=node_n)

    def forward(self, x):
        y = self.gcn(x) 
        return y      


#######################################
class FC_ELU(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0., no_bn=False):
        super(FC_ELU, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.ELU(inplace=True),
            nn.Dropout(do_prob),
            nn.Linear(n_hid, n_out),
            nn.ELU(inplace=True))
        if no_bn:
            self.bn = None
        else:
            self.bn = nn.BatchNorm1d(n_out)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        orig_shape = inputs.shape
        x = inputs.view(-1, inputs.size(-1))
        x = self.bn(x)
        return x.view(orig_shape)

    def forward(self, inputs):
        x = self.model(inputs)
        if self.bn is not None:
            return self.batch_norm(x)
        else:
            return x


#######################################
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot



#######################################
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: adapted from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # based on https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        y_hard = y_hard.to(logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # it achieves two things:
        # - makes the output value exactly one-hot (since we add then subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip all other gradients)
        y = y_hard - y_soft.data + y_soft
    else:
        y = y_soft
    return y


#######################################
def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: adapted from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps).to(logits.device)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


#######################################
def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: adapted from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


#######################################
class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


#######################################
class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


#######################################
def pono(x, epsilon=1e-5):
    """Positional normalization"""
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output, mean, std