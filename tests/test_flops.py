from models.hybrid_model import HybridModel
from utils.flops import print_flops


model = HybridModel()

print_flops(model)