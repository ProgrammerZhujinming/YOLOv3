import torch
source_pre_model_pth = "./weights/Darknet-53_60.pth"
destination_model_pth = "./weights/Darknet-53_final.pth"
param_dict = torch.load(source_pre_model_pth, map_location=torch.device("cpu"))['min_loss_model']
torch.save(param_dict, destination_model_pth)