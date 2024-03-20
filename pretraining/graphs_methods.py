import matplotlib.pyplot as plt

def losses_graph(epoch_info):

    # Plot delle curve di loss di training e validazione
    plt.plot(epoch_info['train_losses'], label='Train Loss')
    plt.plot(epoch_info['val_losses'], label='Val Loss')

    # Aggiungi etichette agli assi e una legenda
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Mostra il grafico
    plt.show()