# traffic_sign_ann.py
import torch
import torch.nn as nn

class TrafficSignCNN_AE_ANN(nn.Module):
    """
    Convolutional Neural Network (CNN) with an Autoencoder (AE) component
    and an Artificial Neural Network (ANN) for classification.
    """
    def __init__(self, device):
        super(TrafficSignCNN_AE_ANN, self).__init__()
        self.device = device

        # CNN Feature Extractor
        # Input size: 3x224x224
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 32x112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 64x56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # Output: 128x28x28
        )

        # Calculate the flattened size after feature_extractor
        # Assuming input 224x224 -> MaxPool2d (x3) -> 128 channels * (224 / 2^3) * (224 / 2^3)
        self._feature_extractor_output_size = 128 * 28 * 28 # 128 * 784 = 100352

        # Encoder (AE component - reduces dimensionality for classification)
        self.encoder = nn.Sequential(
            nn.Linear(self._feature_extractor_output_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # ANN Classifier
        # Takes the encoded features (512 dimensions) as input
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), # Modified: Adjusted Dropout rate for debugging (you can set to 0.0 if preferred)
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(), # Changed ReLU to LeakyReLU here as well for consistency and to mitigate dying ReLUs
            nn.Dropout(0.5), # Modified: Adjusted Dropout rate
            nn.Linear(256, 63)  # 63 classes for multi-label classification
        )

        # Move the entire model to the specified device (CPU/GPU)
        self.to(device)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input image tensor.
        Returns:
            torch.Tensor: Output logits for classification.
        """
        # Ensure input is on the correct device
        x = x.to(self.device)

        # Pass through CNN feature extractor
        x = self.feature_extractor(x) 
        

        # Flatten the output of the feature extractor for the encoder
        x = x.view(x.size(0), -1) # -1 infers the dimension, (batch_size, 128*28*28)


        # Pass through the encoder
        # We can add more granular prints within the encoder if needed
        # For now, let's print before and after encoder's first activation
        x_before_encoder_relu1 = self.encoder[0](x) # Output of first Linear layer
        x_after_encoder_bn1 = self.encoder[1](x_before_encoder_relu1) # Output of first BatchNorm1d
        x_after_encoder_relu1 = self.encoder[2](x_after_encoder_bn1) # Output of first ReLU
        x_before_encoder_relu2 = self.encoder[3](x_after_encoder_relu1) # Output of second Linear layer
        x_after_encoder_bn2 = self.encoder[4](x_before_encoder_relu2) # Output of second BatchNorm1
        encoded = self.encoder[5](x_after_encoder_bn2) # Output of second ReLU, which is the final encoded feature
        


        # Pass through the classifier
        # Similarly, add granular prints within the classifier
        x_after_dropout1 = self.classifier[0](encoded) # After first Dropout
        x_before_classifier_leaky_relu = self.classifier[1](x_after_dropout1) # After first Linear
        x_after_classifier_bn = self.classifier[2](x_before_classifier_leaky_relu) # After BatchNorm1d
        x_after_leaky_relu = self.classifier[3](x_after_classifier_bn) # After LeakyReLU
        x_after_dropout2 = self.classifier[4](x_after_leaky_relu) # After second Dropout
        logits = self.classifier[5](x_after_dropout2) # Final Linear layer output (logits)
        return logits
