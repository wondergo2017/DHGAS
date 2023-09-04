from .GCN import GCN
from .GAT import GAT
from .RGCN import RGCN
from .HGT import HGT
from .DyHATR import DyHATR
from .HTGNN import HTGNN
from .DHSpace import DHSpace, DHNet
from .DHSpaceSearch import DHSearcher
from .load_model import load_model

Sta_MODEL = "GCN GAT RGCN HGT HGT+".split()
Homo_MODEL = "GCN GAT RGCN DySAT".split()
