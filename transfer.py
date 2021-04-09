import torch
import torchvision
from model import U2NET

model_dir = "saved_models/u2net/u2net.pth"
net = U2NET(3, 1)
example = torch.rand(1, 3, 320, 320)

net.load_state_dict(torch.load(model_dir,map_location=torch.device('cpu')))

traced_script_module = torch.jit.trace(net, example)
output = traced_script_module(torch.ones(1, 3, 320, 320))
print(len(output))
traced_script_module.save("U2net_model.pt")