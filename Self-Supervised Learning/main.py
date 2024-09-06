from src.train import train_pretext_task, fine_tune_model, evaluate_model

def main():
    # Step 1: Pretext Task Training
    print("Training on Rotation Prediction Task...")
    model = train_pretext_task(num_epochs=5)

    # Step 2: Fine-Tuning
    print("Fine-tuning the model for Digit Classification...")
    model = fine_tune_model(model, num_epochs=5)

    # Step 3: Evaluation
    print("Evaluating the model...")
    accuracy = evaluate_model(model)
    print(f'Accuracy of the model on the 10,000 test images: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
