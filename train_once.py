from train_model import train_model

if __name__ == '__main__':
    # Start small to test: 3 epochs. Later bump to 30-50.
    train_model(epochs=3, batch_size=128, lr=0.001, save_path='cifar10_cnn.pth')
    print("Saved cifar10_cnn.pth")