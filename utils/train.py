from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt


def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    with tqdm(testloader, position=0, leave=False) as progress_bar:          
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)

            output = model(images)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            predictions = ps.max(dim=1)[1]
            equality = (labels.data == predictions)
            accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


def test(model, testloader, device='cuda'):  
    model.to(device)
    accuracy = 0

    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
                    
            output = model(images)
            
            ps = torch.exp(output)
            predictions = ps.max(dim=1)[1]
            equality = (labels.data == predictions)
            accuracy += equality.type(torch.FloatTensor).mean()

            for t, p in zip(labels.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        
        print('Testing accuracy: {:.3f}'.format(accuracy/len(testloader)))
    return accuracy


def train(model, model_name, batch_size, n_epochs, lr, train_loader, val_loader, saved_model_path, device = "cuda"):
    start_time = datetime.now()

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr= lr)

    train_loss_ls = []
    val_loss_ls = []
    val_acc_ls = []
    train_acc_ls = []
    
    best_accuracy = 0
    best_recall = 0
    
    steps = 0
    train_acc = 0
    
    
    running_loss = 0.0
    running_loss2 = 0.0
    print("Training started")
    for e in range(n_epochs):  # loop over the dataset multiple times

        # Training
        model.train()
        for idx, (images, labels) in enumerate(train_loader) :
            steps += 1
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            ps = torch.exp(output)
            predictions = ps.max(dim=1)[1]
            equality = (labels.data == predictions)
            train_acc += equality.type(torch.FloatTensor).mean()

            # print statistics
            running_loss += loss.item()
            running_loss2 += loss.item()
            

            if steps % validate_every == 0:

                # Eval mode for predictions
                model.eval()

                # Turn off gradients for validation
                with torch.no_grad():
                    test_loss, test_accuracy = validation(model, val_loader, criterion, device)

                running_loss /= validate_every
                train_acc /= validate_every
                print(f"Epoch: {e+1}/{ n_epochs} - Training Loss: {running_loss} - Validation Loss: {test_loss/len(val_loader)} - Train Accuracy:  {train_acc} - Validation Accuracy:  {test_accuracy/len(val_loader)}")

                train_loss_ls.append(running_loss) #/print_every
                val_loss_ls.append(test_loss/len(val_loader))
                val_acc_ls.append(test_accuracy/len(val_loader))
                train_acc_ls.append(train_acc)
                running_loss = 0        
                train_acc = 0        
#                 validate_every = 0
                # Make sure training is back on
                model.train()

            elif  steps % print_every == 0:
                print("Epoch: {}/{} - ".format(e+1, n_epochs), f"- Step: {idx}/{len(train_loader)}" , "Training Loss: {:.3f} - ".format(running_loss2/print_every))
                running_loss2 = 0
                    
        filepath = saved_model_path + f"{model_name}-{start_time}-b{batch_size}-e{e}.pt"
        torch.save(model, filepath)

    print("Finished training")
    
    fig, axs = plt.subplots(2,figsize=(10,15))
    
    axs[0].plot(train_loss_ls, label = "train_loss")
    axs[0].plot(val_loss_ls, label = "val_loss")
    axs[0].legend()
    
    axs[1].plot(train_acc_ls,label = "train_acc")
    axs[1].plot(val_acc_ls,label = "val_acc")
    axs[1].legend()

    fig.savefig(saved_model_path+'history.png')
    fig.show()

    return model.state_dict(), (train_loss_ls, val_loss_ls,val_acc_ls,train_acc_ls)
