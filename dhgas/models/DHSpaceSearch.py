from dhgas.trainer import *
from torch import nn
import torch.nn.functional as F
import torch

import numpy as np
import torch
import math
from torch_scatter import scatter
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, ones
from torch_geometric.nn import GCNConv
from dhgas.models.HLinear import HLinear, HLayerNorm
from pprint import pprint


class DHSpace(nn.Module):
    def __init__(
        self,
        hid_dim,
        metadata,
        twin,
        K_To,
        K_N,
        K_R,
        n_heads=8,
        norm=True,
        causal_mask=False,
        last_mask=False,
        full_mask=False,
        node_hetero=False,
        node_entangle_type="None",
        rel_entangle_type="None",
        rel_time_type="relative",
        time_patch_num=1,
        hupdate=True,
        **kwargs,
    ):
        super(DHSpace, self).__init__()

        # Data Attrs
        self.twin = twin

        # Data Type Info
        self.metadata = metadata
        self.node_types, self.edge_types = metadata
        self.num_types = len(self.node_types)
        self.num_relations = len(self.edge_types)
        self.ntype2id = dict(zip(self.node_types, range(self.num_types)))
        self.etype2id = dict(
            zip([et for _, et, _ in self.edge_types], range(self.num_relations))
        )  # [etype] -> [0,1,2]
        self.id2ntype = self.node_types
        # [0,1,2] -> [(ntype_src,etype,ntype_tar)]
        self.id2etype = self.edge_types

        # Space Constraint
        self.K_To = K_To
        self.K_N = K_N
        self.K_R = K_R
        self.causal_mask = causal_mask
        self.last_mask = last_mask
        self.full_mask = full_mask
        self.node_hetero = node_hetero
        self.norm = norm
        self.rel_time_type = rel_time_type
        self.collect_att = True
        self.time_patch_num = time_patch_num
        self.hupdate = hupdate

        # Model hyperparams
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.d_k = hid_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.aggr = "add"
        self.node_dim = 0

        # init
        self.dummy_A_To = torch.ones(
            (twin, twin, self.num_relations)
        ).long()  # T x T x C_e
        if self.causal_mask:
            for i in range(twin):
                for j in range(i + 1, twin):
                    for k in range(self.num_relations):
                        self.dummy_A_To[i, j, k] = 0
        self.dummy_A_N = torch.zeros((twin, self.num_types)).long()  # T x C_n

        if self.rel_time_type == "independent":
            rel_time_len = twin * twin
        elif self.rel_time_type == "relative":
            # rel_time_len = 2*twin
            if self.causal_mask:
                rel_time_len = twin
            else:
                rel_time_len = 2 * twin

        elif self.rel_time_type == "source":
            rel_time_len = twin
        elif self.rel_time_type == "target":
            rel_time_len = twin
        else:
            raise NotImplementedError(f"Unknown rel_time_type {rel_time_type}")
        # rel_time_len = twin*twin # test : force all include

        self.dummy_A_R = torch.zeros(
            (rel_time_len, self.num_relations)
        ).long()  # len x T x C_e
        self.assign_arch(self.get_A_init())
        self.init_supernet()

        # fix
        self.fix_N = False
        self.fix_R = False
        self.fix_To = True

    def init_supernet(self):
        hid_dim = self.hid_dim
        twin = self.twin
        K_To = self.K_To
        K_N = self.K_N
        K_R = self.K_R
        n_heads = self.n_heads
        d_k = self.d_k

        # Message
        def makelinear(n):
            linears = nn.ModuleList()
            for ntype in range(n):
                linears.append(nn.Linear(hid_dim, hid_dim))
            return linears

        self.k_linears = makelinear(K_N)
        self.q_linears = makelinear(K_N)
        self.v_linears = makelinear(K_N)

        # Aggregate
        self.relation_pri = nn.Parameter(torch.ones(K_R, n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(K_R, n_heads, d_k, d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(K_R, n_heads, d_k, d_k))

        ones(self.relation_pri)
        glorot(self.relation_att)
        glorot(self.relation_msg)

        if self.hupdate:
            self.update_lin = HLinear(hid_dim, self.metadata, act="None")

        if self.norm:
            self.update_norm = HLayerNorm(hid_dim, self.metadata)

        self.n_alpha = nn.Parameter(
            torch.randn((self.time_patch_num * self.num_types, K_N)) * 1e-3
        )
        self.r_alpha = nn.Parameter(
            torch.randn((self.time_patch_num * self.num_relations, K_R)) * 1e-3
        )
        self.to_alpha = nn.Parameter(
            torch.randn((self.twin, self.twin, self.num_relations)) * 1e-3
        )

    def reset_parameters(self, atypes=None):
        if atypes == None:
            atypes = "node rel update norm nalpha ralpha toalpha".split()

        for atype in atypes:
            assert atype in "node rel update norm nalpha ralpha toalpha".split()

        if "node" in atypes:
            for ls in [self.k_linears, self.q_linears, self.v_linears]:
                for l in ls:
                    l.reset_parameters()
        if "rel" in atypes:
            ones(self.relation_pri)
            glorot(self.relation_att)
            glorot(self.relation_msg)
        if "update" in atypes:
            if self.hupdate:
                self.update_lin.reset_parameters()
        if "norm" in atypes:
            if self.norm:
                self.update_norm.reset_parameters()
        if "nalpha" in atypes:
            self.n_alpha.data = (
                torch.randn((self.time_patch_num * self.num_types, self.K_N)) * 1e-3
            ).to(self.n_alpha.device)
        if "ralpha" in atypes:
            self.r_alpha.data = (
                torch.randn((self.time_patch_num * self.num_relations, self.K_R)) * 1e-3
            ).to(self.r_alpha.device)
        if "toalpha" in atypes:
            self.to_alpha.data = (
                torch.randn((self.twin, self.twin, self.num_relations)) * 1e-3
            ).to(self.to_alpha.device)

    def get_A_init(self):
        return (self.dummy_A_To.clone(), self.dummy_A_N.clone(), self.dummy_A_R.clone())

    def get_arch(self):
        return [a.clone() for a in self.A]

    def assign_arch(self, A):
        self.A = A
        return self

    def assign_basic_arch(self, atype):
        return self.assign_arch(self.basic_space(atype))

    def setATo(self, atype="full"):
        twin = self.twin
        num_relations = self.num_relations
        if atype == "full":
            ATo = self.A[0]
            for i in range(twin):
                for j in range(twin):
                    for k in range(num_relations):
                        ATo[i, j, k] = 1
            if self.causal_mask:
                for i in range(twin):
                    for j in range(i + 1, twin):
                        for k in range(num_relations):
                            ATo[i, j, k] = 0
            if self.last_mask:
                for i in range(twin - 1):
                    for j in range(twin):
                        for k in range(num_relations):
                            ATo[i, j, k] = 0
        else:
            raise NotImplementedError(f"Unknown ATo atype {atype}")

    def setAN(self, atype="same"):
        twin = self.twin
        num_types = self.num_types
        patch_num = self.time_patch_num
        part = int(np.ceil(twin / patch_num))
        if atype == "same":
            AN = self.A[1]
            for i in range(twin):
                for j in range(num_types):
                    AN[i, j] = 0
        elif atype == "hetero":
            AN = self.A[1]
            for i in range(twin):
                for j in range(num_types):
                    AN[i, j] = j
        elif atype == "t-hetero":
            AN = self.A[1]
            for i in range(patch_num):
                for j in range(num_types):
                    AN[i * part : i * part + part, j] = i * num_types + j
        else:
            raise NotImplementedError(f"Unknown ATo atype {atype}")

    def setAR(self, atype="same"):
        twin = self.twin
        num_relations = self.num_relations
        patch_num = self.time_patch_num
        part = int(np.ceil(twin / patch_num))
        if atype == "same":
            AR = self.A[2]
            for i in range(twin):
                for j in range(num_relations):
                    AR[i, j] = 0
        elif atype == "hetero":
            AR = self.A[2]
            for i in range(twin):
                for j in range(num_relations):
                    AR[i, j] = j
        elif atype == "t-hetero":
            AR = self.A[2]
            for i in range(patch_num):
                for j in range(num_relations):
                    AR[i * part : i * part + part, j] = i * num_relations + j
        else:
            raise NotImplementedError(f"Unknown ATo atype {atype}")

    def basic_space(self, atype):
        ATo, AN, AR = self.get_A_init()
        twin = self.twin
        num_relations = self.num_relations
        num_types = self.num_types
        # default all connect and shared.
        if "full" in atype:
            ATo = ATo.new_ones(ATo.shape)
        if "last" in atype:
            ATo = ATo.new_zeros(ATo.shape)
            ATo[-1, :, :] = 1
        if "causal" in atype:
            ATo = ATo.new_ones(ATo.shape)
            for i in range(twin):
                for j in range(i + 1, twin):
                    for k in range(num_relations):
                        ATo[i, j, k] = 0
        if "node_hetero" in atype:
            AN = AN.new_zeros(AN.shape)
            for i in range(twin):
                for j in range(num_types):
                    AN[i, j] = j
        if "rel_hetero" in atype:
            AR = AR.new_zeros(AR.shape)
            for i in range(2 * twin):
                for j in range(num_relations):
                    AR[i, j] = j
        return (ATo, AN, AR)

    def count_space(self):
        ATo, AN, AR = self.A
        kto = ATo.sum().item()
        kn = len(AN.unique())
        kr = len(AR.unique())
        return kto, kn, kr

    @torch.no_grad()
    def fix_n_alpha(self):
        AN = self.A[1]
        for i in range(AN.shape[0]):
            for j in range(AN.shape[1]):
                alpha = self.n_alpha[AN[i, j]]
                AN[i, j] = torch.argmax(alpha).item()

    @torch.no_grad()
    def fix_r_alpha(self):
        AR = self.A[2]
        for i in range(AR.shape[0]):
            for j in range(AR.shape[1]):
                alpha = self.r_alpha[AR[i, j]]
                AR[i, j] = torch.argmax(alpha).item()

    @torch.no_grad()
    def fix_to_alpha(self):
        K_To = self.K_To
        to_alpha = self.to_alpha  # [T,T,num_relations]
        twin = self.twin
        num_relations = self.num_relations
        ATo = self.A[0].clone()
        for t in range(twin):
            for i in range(twin):
                for r in range(num_relations):
                    self.A[0][t, i, r] = 0

        for t in range(twin):
            valid_alpha = []
            for i in range(twin):
                for r in range(num_relations):
                    if ATo[t, i, r]:
                        valid_alpha.append([to_alpha[t, i, r].item(), i, r])
            valid_alpha = sorted(valid_alpha, key=lambda x: x[0], reverse=True)
            soft_alpha = torch.FloatTensor([x[0] for x in valid_alpha])
            soft_alpha = F.softmax(soft_alpha, dim=0)
            for i, a in enumerate(soft_alpha):
                valid_alpha[i][0] = np.round(a.item(), 4)

            pprint(f"Time {t} attn:")
            pprint(valid_alpha)
            budget = K_To
            if not self.causal_mask:
                budget = min(budget, (t + 1) * self.num_relations)
            chosen_alpha = valid_alpha[:budget]
            for att, i, r in chosen_alpha:
                self.A[0][t, i, r] = 1

    def set_stage(self, stage):
        print(f'{"#"*10} stage {stage} {"#"*100}')
        if stage == 0:
            self.setATo("full")
            self.fix_To = True

            self.setAN("same")
            self.fix_N = True

            self.setAR("same")
            self.fix_R = True

        elif stage == 1:
            self.setATo("full")
            self.fix_To = True

            self.setAN("t-hetero")
            self.fix_N = False

            self.setAR("same")
            self.fix_R = True

        elif stage == 2:
            self.setATo("full")
            self.fix_To = True

            self.fix_n_alpha()
            self.fix_N = True

            self.setAR("t-hetero")
            self.fix_R = False

        elif stage == 3:
            self.setATo("full")
            self.fix_To = False

            self.fix_N = True

            self.fix_r_alpha()
            self.fix_R = True

        elif stage == 4:
            self.fix_to_alpha()
            self.fix_To = True
            self.fix_N = True
            self.fix_R = True

        else:
            raise NotImplementedError(f"Unknown stage {stage}")

    def get_modules_weights(self, target_type, source_type, ttar, tsrc, relation_type):
        _, AN, AR = self.A
        source_type = self.ntype2id[source_type]
        target_type = self.ntype2id[target_type]
        relation_type = self.etype2id[relation_type]
        FNsrc = AN[tsrc, source_type]
        FNtar = AN[ttar, target_type]
        if self.rel_time_type == "relative":
            dt = tsrc - ttar + self.twin - 1
        elif self.rel_time_type == "independent":
            dt = ttar * self.twin + tsrc
        elif self.rel_time_type == "source":
            dt = tsrc
        elif self.rel_time_type == "target":
            dt = ttar
        else:
            raise NotImplementedError(f"Unknown rel_time_type {self.rel_time_type}")

        def n_combine(funcs, alpha):
            def aggfunc(*args, **kwargs):
                x = torch.stack([func(*args, **kwargs) for func in funcs])
                alpha_shape = [-1] + [1] * (len(x.size()) - 1)
                # import pdb;pdb.set_trace()
                x = torch.sum(x * F.softmax(alpha, -1).view(*alpha_shape), 0)
                return x

            return aggfunc

        def r_combine(funcs, alpha):
            x = funcs
            alpha_shape = [-1] + [1] * (len(x.size()) - 1)
            x = torch.sum(x * F.softmax(alpha, -1).view(*alpha_shape), 0)
            return x

        if self.fix_N:
            q_linear = self.q_linears[FNtar]
            k_linear = self.k_linears[FNsrc]
            v_linear = self.v_linears[FNsrc]
        else:
            q_linear = n_combine(self.q_linears, self.n_alpha[FNtar])
            k_linear = n_combine(self.k_linears, self.n_alpha[FNsrc])
            v_linear = n_combine(self.v_linears, self.n_alpha[FNsrc])

        FR = AR[dt, relation_type]

        if self.fix_R:
            relation_att = self.relation_att[FR]
            relation_pri = self.relation_pri[FR]
            relation_msg = self.relation_msg[FR]
        else:
            relation_att = r_combine(self.relation_att, self.r_alpha[FR])
            relation_pri = r_combine(self.relation_pri, self.r_alpha[FR])
            relation_msg = r_combine(self.relation_msg, self.r_alpha[FR])

        return q_linear, k_linear, v_linear, relation_att, relation_pri, relation_msg

    def DHAttn(self, x_tar_rel, x_src_rel, t_tar, t_src, rel):
        """x_tar(ttar) <- x_src(tsrc) with eindex at tsrc and relation_type
        Note that (j,i) -> (src,target)
        @return att [E,h] ; msg [E,h,F/h]
        """
        # to sparse
        target_node_vec = x_tar_rel
        source_node_vec = x_src_rel

        # get weights
        source_type, _, target_type = self.id2etype[self.etype2id[rel]]
        (
            q_linear,
            k_linear,
            v_linear,
            relation_att,
            relation_pri,
            relation_msg,
        ) = self.get_modules_weights(target_type, source_type, t_tar, t_src, rel)

        # calculation
        q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
        k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
        v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)  # [E,h,F/h]

        k_mat = torch.bmm(k_mat.transpose(1, 0), relation_att).transpose(1, 0)
        res_att = (q_mat * k_mat).sum(dim=-1) * relation_pri / self.sqrt_dk  # [E,h]
        res_msg = torch.bmm(v_mat.transpose(1, 0), relation_msg).transpose(
            1, 0
        )  # [E,h,F/h]
        return res_att, res_msg

    def DHAttnOne2Multi(self, x_tar, topos):
        # message
        res_atts = {}
        res_msgs = {}
        ei_tars = {}
        to_weights = {}
        for (x_tar_rel, x_src_rel, t_tar, t_src, rel, ei_rel_tar) in topos:
            # att [E,h] , v [E,h,F/h]
            _, _, target_type = self.id2etype[self.etype2id[rel]]
            for collect in [res_atts, res_msgs, ei_tars, to_weights]:
                if target_type not in collect:
                    collect[target_type] = []
            att, msg = self.DHAttn(x_tar_rel, x_src_rel, t_tar, t_src, rel)

            res_atts[target_type].append(att)  # attention of : ntar(ttar) -> T x rel
            res_msgs[target_type].append(msg)  # message of : ntar(ttar) -> T x rel
            ei_tars[target_type].append(
                ei_rel_tar
            )  # target index of : E(ntar(ttar)) -> T x rel

            if not self.fix_To:
                to_weights[target_type].append(
                    self.to_alpha[t_tar, t_src, self.etype2id[rel]].expand(att.shape)
                )

        x_dict = {}
        for ntype in self.node_types:
            if ntype in res_atts:
                # aggregate
                res_att = torch.cat(res_atts[ntype], dim=0)
                res_msg = torch.cat(res_msgs[ntype], dim=0)
                ei_tar = torch.cat(ei_tars[ntype])

                res_att = softmax(res_att, ei_tar)  # [ET,h]
                if not self.fix_To:
                    to_weight = torch.cat(to_weights[ntype], dim=0)
                    res_att = res_att.mul(softmax(to_weight, ei_tar))  # element wise
                    del to_weights[ntype]

                res = res_msg * res_att.view(-1, self.n_heads, 1)  # [E,h,F/h]

                del res_atts[ntype], res_msgs[ntype]

                res = res.view(-1, self.hid_dim)  # [E,F]
                res = scatter(
                    res,
                    ei_tar,
                    dim=self.node_dim,
                    dim_size=x_tar[ntype].shape[0],
                    reduce=self.aggr,
                )  # [N,F]
                del ei_tars[ntype]

                # update
                if self.hupdate:
                    res = self.update_lin[ntype](F.gelu(res))
                # res = self.update_lin(F.gelu(res))
                res = x_tar[ntype] + res
            else:
                # aggregate nothing
                # update
                res = x_tar[ntype]
            x_dict[ntype] = res
        return x_dict

    def forward(self, xs, graphs):
        # hyper
        twin = self.twin
        device = xs[0][self.id2ntype[0]].device

        # data
        x_win = xs  # [time,ntype] -> [N,F]
        x_res = []  # [time,ntype] -> [N,F]

        # calculation
        ATo, _, _ = self.A
        ATo = ATo.to(device)
        for t_tar in range(twin):
            # get topo need to process
            ATo_tar = ATo[t_tar]
            if ATo_tar.sum() == 0:  # no need : use original
                x_dict = x_win[t_tar]
            else:
                topos = []
                x_tar = x_win[t_tar]
                # select all attention in need of computation w.r.t topology
                for t_src in range(twin):
                    if ATo_tar[t_src].sum() == 0:
                        continue
                    graph_src = graphs[t_src]
                    x_src = x_win[t_src]
                    for rel in ATo_tar[t_src].nonzero():
                        nsrc, rel, ntar = self.id2etype[rel]
                        ei_rel = graph_src[rel].edge_index
                        x_tar_rel = x_tar[ntar].index_select(
                            self.node_dim, ei_rel[1, :]
                        )
                        x_src_rel = x_src[nsrc].index_select(
                            self.node_dim, ei_rel[0, :]
                        )
                        ei_rel_tar = ei_rel[1, :].T
                        topo = (x_tar_rel, x_src_rel, t_tar, t_src, rel, ei_rel_tar)
                        topos.append(topo)
                # calculate
                x_dict = self.DHAttnOne2Multi(x_tar, topos)
                if self.norm:
                    x_dict = self.update_norm(x_dict)
            x_res.append(x_dict)
        return x_res

    def __repr__(self):
        return f"""{super().__repr__()} with 
        metadict : {self.node_types,self.edge_types}
        Cn,Ce,T : {self.num_types,self.num_relations,self.twin}
        KTo,KN,KR : {self.K_To,self.K_N,self.K_R}
        hiddim,n_heads,d_k : {self.hid_dim,self.n_heads,self.d_k}
        A : {self.A}
        """


def HAct(x_dict, act=F.relu):
    if isinstance(x_dict, list):
        for x in x_dict:
            HAct(x, act)
    else:
        for ntype in x_dict:
            x_dict[ntype] = act(x_dict[ntype])
    return x_dict


class DHNet(nn.Module):
    def __init__(
        self,
        hid_dim,
        twin,
        metadata,
        spaces,
        predict_type,
        featemb=None,
        nclf_linear=None,
        hlinear_act="tanh",
        agg="last",
    ):
        super(DHNet, self).__init__()

        # Data Attr
        self.hid_dim = hid_dim
        self.metadata = metadata
        self.twin = twin
        self.predict_type = predict_type

        # Searchable Archs
        self.spaces = nn.ModuleList(spaces)

        # Fixed Backbone
        self.hlinear = HLinear(hid_dim, metadata, act=hlinear_act)
        self.featemb = featemb if featemb else lambda x: x
        self.act = lambda x: HAct(x, F.relu)
        self.nclf_linear = nclf_linear
        self.agg = agg

    def reset_parameters(self, tp=None):
        if not tp:
            tp = ["space", "hlinear", "featemb"]
        for t in tp:
            if t not in ["space", "hlinear", "featemb"]:
                raise NotImplementedError(f"Unknown type {t}")
        if "space" in tp:
            for space in self.spaces:
                space.reset_parameters()
        if "hlinear" in tp:
            self.hlinear.reset_parameters()
        if "featemb" in tp:
            self.featemb.reset_parameters()

    def freeze(self, module_names):
        for name in module_names:
            assert name in "feat_emb hlinear hupdate hnorm rel".split(), print(
                f"Unimplemented {module_names}"
            )

        if "feat_emb" in module_names:
            for param in self.featemb.parameters():
                param.requires_grad = False
        if "hlinear" in module_names:
            for param in self.hlinear.parameters():
                param.requires_grad = False

        if "hupdate" in module_names:
            for space in self.spaces:
                if space.hupdate:
                    for param in space.update_lin.parameters():
                        param.requires_grad = False

        if "hnorm" in module_names:
            for space in self.spaces:
                if space.norm:
                    for param in space.update_norm.parameters():
                        param.requires_grad = False

        if "rel" in module_names:
            for space in self.spaces:
                for rel in [space.relation_att, space.relation_pri, space.relation_msg]:
                    rel.requires_grad = False

    def assign_arch(self, As):
        for i, space in enumerate(self.spaces):
            space.assign_arch(As[i])
        return self

    def get_configs(self):
        return [space.get_arch() for space in self.spaces]

    def count_space(self):
        ks = np.array([0, 0, 0])
        for space in self.spaces:
            k = np.array(list(space.count_space()))
            ks += k
        return list(map(int, ks))

    def encode(self, graphs, *args, **kwargs):
        xs = []  # [time,ntype] -> [N,F]
        for graph in graphs:
            x_dict = self.featemb(graph.x_dict)
            x_dict = self.hlinear(x_dict)
            xs.append(x_dict)
        out = xs
        for i, conv in enumerate(self.spaces):
            out = conv(out, graphs)
            if i != len(self.spaces) - 1:
                out = self.act(out)

        predict_type = self.predict_type
        if self.agg == "last":
            out = out[-1]
            if isinstance(predict_type, list):
                out = [out[predict_type[0]], out[predict_type[1]]]
            else:
                out = out[predict_type]
        elif self.agg == "sum":
            out = out[-1]
            if isinstance(predict_type, list):
                out = [
                    sum(x[predict_type[0]] for x in out),
                    sum(x[predict_type[1]] for x in out),
                ]
            else:
                out = sum(x[predict_type] for x in out)
        return out

    def decode(self, z, edge_label_index):
        if isinstance(z, list) or isinstance(z, tuple):
            return (z[0][edge_label_index[0]] * z[1][edge_label_index[1]]).sum(dim=-1)
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_nclf(self, z):
        out = self.nclf_linear(z)
        return out


from dhgas.utils import EarlyStopping
import os
from tqdm import tqdm
from dhgas.trainer import load_train_test


class DHSearcher:
    def __init__(
        self,
        criterion,
        save_dir,
        writer,
        n_warmup=100,  # 1000
        disable_progress=False,
        gradient_clip=5.0,
        arch_lr=3e-3,
        arch_wd=1e-3,
        args=None,
    ):
        self.n_warmup = n_warmup
        self.disable_progress = disable_progress
        self.criterion = criterion
        self.save_dir = save_dir
        self.writer = writer
        self.gradient_clip = gradient_clip
        self.model_lr = args.lr
        self.model_wd = args.wd
        self.arch_lr = arch_lr
        self.arch_wd = arch_wd
        self.args = args

    def search(self, net, spaces, dataset, topk=1):
        self.net = net
        self.spaces = spaces
        self.dataset = dataset
        args = self.args

        self.arch_optim = torch.optim.Adam(
            [p for n, p in net.named_parameters() if "alpha" in n],
            lr=self.arch_lr,
            weight_decay=self.arch_wd,
        )
        self.model_optim = torch.optim.Adam(
            [p for n, p in net.named_parameters() if "alpha" not in n],
            lr=self.model_lr,
            weight_decay=self.model_wd,
        )

        self.stage_training()
        config = self.net.get_configs()
        config = [config[0]] * (args.n_layers - 1) + [config[1]]
        return [[config, 0]]

    def stage_training(self):
        args = self.args
        reset_type = args.reset_type
        reset_type2 = args.reset_type2
        patience = args.patience
        spaces = self.spaces
        n_warmup = self.n_warmup
        print(f"reset,patience,n_warmup:{reset_type,patience,n_warmup}")
        # set all to fixed
        for space in spaces:
            space.set_stage(0)

        print(f"# space layers : {len(spaces)} ")
        # sequentially set to search
        for i, space in enumerate(spaces):
            # if i in [0,len(spaces)-1]:
            if i in [len(spaces)]:
                print(f" jump space at {i}")
                continue
            print(f'{"#"*10} Searching space {i} {"#"*10} ')
            space.set_stage(1)
            if reset_type2 == 0:  # do not reset
                pass
            elif reset_type2 == 1:  # reset all
                space.reset_parameters()
            elif reset_type2 == 2:  # reset only rel
                space.reset_parameters("rel ".split())
            elif reset_type2 == 3:  # reset only node
                space.reset_parameters("node ".split())
            elif reset_type2 == 4:  # reset node and rel
                space.reset_parameters("node rel".split())
            self._train(n_warmup, patience)

            space.set_stage(2)
            if reset_type == 0:  # do not reset
                pass
            elif reset_type == 1:  # reset all
                space.reset_parameters()
            elif reset_type == 2:  # reset only rel
                space.reset_parameters("rel ".split())
            elif reset_type == 3:  # reset only node
                space.reset_parameters("node ".split())
            elif reset_type == 4:  # reset node and rel
                space.reset_parameters("node rel".split())
            self._train(n_warmup, patience)

            space.set_stage(3)
            self._train(n_warmup, patience)
            space.set_stage(4)

    @torch.no_grad()
    def get_n_alpha(self):
        choices = [list(space.A[1].unique().numpy()) for space in self.spaces]
        alphas = [space.n_alpha[choices[i]] for i, space in enumerate(self.spaces)]
        alphas = [F.softmax(alpha, -1).detach().cpu() for alpha in alphas]
        dalpha = [torch.topk(a, k=2, dim=-1)[1] for a in alphas]
        print("N alpha:")
        pprint(alphas)
        pprint(dalpha)

    @torch.no_grad()
    def get_r_alpha(self):
        choices = [list(space.A[2].unique().numpy()) for space in self.spaces]
        alphas = [space.r_alpha[choices[i]] for i, space in enumerate(self.spaces)]
        alphas = [F.softmax(alpha, -1).detach().cpu() for alpha in alphas]
        dalpha = [torch.topk(a, k=2, dim=-1)[1] for a in alphas]
        print("R alpha:")
        pprint(alphas)
        pprint(dalpha)

    @torch.no_grad()
    def get_to_alpha(self):
        print("To alpha:")
        pprint([space.to_alpha.detach().cpu() for space in self.spaces])

    def _train(self, max_epochs, patience):
        writer = self.writer
        args = self.args
        train, test = load_train_test(args)
        if args.dataset == "covid":
            earlystop = EarlyStopping(mode="min", patience=patience)
            best_val_auc = final_test_auc = best_epoch = 1e8
        else:
            earlystop = EarlyStopping(mode="max", patience=patience)
            best_val_auc = final_test_auc = best_epoch = 0

        with tqdm(range(max_epochs), disable=self.disable_progress) as bar:
            for i in bar:
                # phase 1. architecture step
                if args.dataset in "Aminer Ecomm".split():
                    loss = train(
                        self.net,
                        self.arch_optim,
                        self.criterion,
                        [self.dataset.val_dataset],
                    )
                elif args.dataset in "Yelp-nc".split():
                    loss = train(
                        self.net,
                        self.arch_optim,
                        self.criterion,
                        self.dataset.val_dataset,
                    )
                elif args.dataset in "covid ".split():
                    loss = train(
                        self.net,
                        self.arch_optim,
                        self.criterion,
                        self.dataset.val_dataset,
                    )
                # loss = train(self.net, self.arch_optim , self.criterion, [self.dataset.val_dataset])

                # auc = test(self.net, self.dataset.val_dataset)
                # self.get_n_alpha()
                # self.get_r_alpha()
                # self.get_to_alpha()

                # phase 2: child network step
                loss = train(
                    self.net,
                    self.model_optim,
                    self.criterion,
                    self.dataset.train_dataset,
                    self.gradient_clip,
                )

                val_auc = test(self.net, self.dataset.val_dataset)
                test_auc = test(self.net, self.dataset.test_dataset)

                if (val_auc > best_val_auc and args.dataset != "covid") or (
                    val_auc < best_val_auc and args.dataset == "covid"
                ):
                    best_val_auc = val_auc
                    final_test_auc = test_auc
                    best_epoch = i
                    print(f"best val auc : {val_auc} , saving at epoch {i}")
                    torch.save(
                        self.net.state_dict(),
                        os.path.join(self.save_dir, f"checkpoint"),
                    )

                bar.set_postfix(
                    loss=loss,
                    val_auc=val_auc,
                    test_auc=test_auc,
                    bval=best_val_auc,
                    btest=final_test_auc,
                    bepoch=best_epoch,
                )

                if writer:
                    writer.add_scalar("Supernet/train_loss", loss, i)
                    writer.add_scalar("Supernet/val_auc", val_auc, i)
                    writer.add_scalar("Supernet/test_auc", test_auc, i)

                if earlystop.step(val_auc):
                    torch.save(
                        self.net.state_dict(),
                        os.path.join(self.save_dir, f"checkpoint{i}"),
                    )
                    break
        print("loading best val checkpoint")
        self.net.load_state_dict(torch.load(os.path.join(self.save_dir, f"checkpoint")))
