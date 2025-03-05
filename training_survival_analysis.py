
from models import MinimalisticNetwork

import torch.nn
import torch.optim
import torch.utils.data.dataloader
from evaluation import concordance_index_censored, PartialLogLikelihood, PartialMSE
import logging


def get_model(params):
    model_name = params["model"]
    if model_name == "minimalistic_network":
        model = MinimalisticNetwork(input_dim=params["input_dim"], inner_dim=params['inner_dim']).to(params["device"])
    else:
        raise ValueError("Model not found")
    return model

def train_model(dataset, params, writer=None, **kwargs):

    gen = torch.Generator().manual_seed(42)
    torch.manual_seed(42)
    train_ind = int(len(dataset) * 0.95)
    
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_ind, len(dataset) - train_ind],
                                                                generator=gen)
    dataloader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=params["batch_size"],
                                                        shuffle=True, drop_last=False)
    dataloader_test = torch.utils.data.dataloader.DataLoader(test_dataset, batch_size=len(test_dataset), drop_last=False)

    # print(f"Train: {len(train_dataset)} - Test: {len(test_dataset)}")
    # print(f"Dataloader Train: {len(dataloader)} - Test: {len(dataloader_test)}")
    model = get_model(params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    loss_fn = params["loss_fn"]
    losses = []
    losses_test = []
    c_indices = []
    c_indices_test = []
    for epoch in range(params["epochs"]):
        loss_train = 0
        c_index = 0
        model.train()
        for X, y_time, y_event in dataloader:
            X = X.to(params["device"])
            y_time = y_time.to(params["device"])
            y_event = y_event.to(params["device"])
            optimizer.zero_grad()
            y_pred = model(X) #, encoder_combination=params["encoder_combination"], regularize=params["regularize_output"])
            y_pred = torch.flatten(y_pred, 0)
            
            # Only calculate the loss if there is non-censored data
            if y_event.sum() > 1:
                #loss = PartialLogLikelihood(y_pred, y_event, y_time, ties="noties")
                loss = loss_fn(y_pred,y_event, y_time)
                loss.backward()
                loss_train += loss.item()
                optimizer.step()
                #check for nan
                if torch.isnan(y_pred).any():
                    print("Nan-Event")
                    print(params)
                    # print y and corresponding X values that cause nan
                    print(torch.isnan(y_pred).sum())
                            

                else:
                    y_pred = y_pred.detach().cpu().numpy()
                    c_index += concordance_index_censored(y_event.detach().cpu().numpy().astype(bool),
                                                          y_time.detach().cpu().numpy(), y_pred)[0]
                
        # if epoch % 10 == 0:
        #     print(f"Epoch {epoch} - Loss: {loss_train / len(dataloader)}" +
        #           f"- C-Index: {c_index / len(dataloader)}" +
        #           f"- mean y_pred: {y_pred.mean()}")


        losses.append(loss_train / len(dataloader))
        c_indices.append(c_index / len(dataloader))
        if writer is not None:
            writer.add_scalar("Loss/train", loss_train / len(dataloader), epoch)
            writer.add_scalar("C-Index/train", c_index / len(dataloader), epoch)

        loss_test = 0
        c_index = 0
        model.eval()
        if epoch % 10 == 0:
            for X, y_time, y_event in dataloader_test:
                X = X.to(params["device"])
                y_pred = model(X).detach().cpu()
                y_pred = torch.flatten(y_pred, 0)
                y_time = y_time
            
                #loss = PartialLogLikelihood(y_pred, y_event, y_time, ties="noties")
                loss = loss_fn(y_pred,y_event, y_time)
                loss_test += loss.item()
                # c_index += concordance_index_censored(y_event.numpy().astype(bool), y_time.numpy(), y_pred.numpy())[0]
                # modified
                try:
                    c_val = concordance_index_censored(
                                y_event.numpy().astype(bool),
                                y_time.numpy(),
                                y_pred.numpy()
                            )[0]
                except ValueError as e:
                    if str(e) == "All samples are censored":
                        c_val = 0
                    else:
                        raise e
                c_index += c_val



            # print(f"Epoch {epoch} - Eval Loss: {loss_test / len(dataloader_test)}" +
            #     f"- Eval C-Index: {c_index / len(dataloader_test)}")

        losses_test.append(loss_test / len(dataloader_test))
        c_indices_test.append(c_index / len(dataloader_test))
        if writer is not None:
            writer.add_scalar("Loss/test", loss_test / len(dataloader_test), epoch)
            writer.add_scalar("C-Index/test", c_index / len(dataloader_test), epoch)
        # scheduler.step()

    test_eval = {"loss": losses_test, "c_index": c_indices_test}
    return model, losses, test_eval
