import matplotlib.pyplot as plt
import re


def extract_accuracies(log_file_path):
    train_accs = [] 
    val_accs = []  
    train_epochs = []  
    val_epochs = []    


    train_pattern = re.compile(r'\[Train\]\s+Epoch\s+(\d+).*Train_Acc\s+([\d.]+)')
    val_pattern = re.compile(r'\[Val\]\s+Epoch\s+(\d+).*Val_Acc\s+([\d.]+)')

   
    with open(log_file_path, 'r') as file:
        for line in file:
          
            train_match = train_pattern.search(line)
            if train_match:
                epoch, acc = train_match.groups()
                train_accs.append(float(acc))
                train_epochs.append(int(epoch))

         
            val_match = val_pattern.search(line)
            if val_match:
                epoch, acc = val_match.groups()
                val_accs.append(float(acc))
                val_epochs.append(int(epoch))

    
    if len(train_accs) != len(val_accs) or len(train_epochs) != len(val_epochs):
        raise ValueError("Mismatch in number of Train and Val entries")

    
    epochs = train_epochs 

    return train_accs, val_accs, epochs


def plot_accuracies(train_accs, val_accs, epochs):
    plt.figure(figsize=(10, 6))

    
    plt.plot(epochs, train_accs, label='Train Accuracy', marker='o', color='blue')
    plt.plot(epochs, val_accs, label='Validation Accuracy', marker='s', color='red')

  
    plt.title('Train and Validation Accuracy Over Epochs (Combined Runs)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()

    
    plt.xticks(epochs)

  
    plt.savefig('/kaggle/working/results/run_resnet50_imagenet_prune1/accuracy_plot.png')
    plt.show()


log_file_path = '/kaggle/working/results/run_resnet50_imagenet_prune1/train_logger.log'


try:
    train_accs, val_accs, epochs = extract_accuracies(log_file_path)
    print("Extracted data:")
    print("Train Accuracies:", train_accs)
    print("Validation Accuracies:", val_accs)
    print("Epochs:", epochs)
    plot_accuracies(train_accs, val_accs, epochs)
except Exception as e:
    print(f"Error: {e}")
