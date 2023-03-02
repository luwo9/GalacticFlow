import numpy as np
import torch
import flowcode
import time


train_data = torch.load("data_cond_trainer.pth")
model = torch.load("model_cond_trainer.pth").to("cuda")
train_parameters = np.load("params_cond_trainer.npy")
filename = str(np.load("filename_cond_trainer.npy"))

train_loss_saver = []

start = time.perf_counter()

flowcode.train_flow(model, train_data, train_parameters[:-3].astype("int"), int(train_parameters[-3]), lr=train_parameters[-2], batch_size=int(train_parameters[-1]), loss_saver=train_loss_saver)

end = time.perf_counter()

torch.save(model.state_dict(), f"{filename}.pth")
np.save(f"loss_{filename}.npy",np.array(train_loss_saver+[end-start]))
