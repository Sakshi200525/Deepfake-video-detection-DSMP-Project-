import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class DeepfakeDetectionModel(nn.Module):
    """
    Advanced multi-modal deepfake detection model
    Combines visual, audio, and temporal features
    """

    def __init__(self, num_classes=2, dropout=0.5):
        super(DeepfakeDetectionModel, self).__init__()

        # 1. Visual feature extractor (EfficientNet-B4)
        # We load a pre-trained model and remove its final classification layer.
        self.visual_backbone = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
        visual_features = self.visual_backbone.classifier[
            1].in_features  # Get the size of features before the final layer
        self.visual_backbone.classifier = nn.Identity()  # Replace the classifier with an identity layer (no-op)

        # 2. Audio feature extractor (CNN)
        # Processes the mel spectrogram (size 1x128x128)
        self.audio_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Reduces feature map to 128 features total
        )

        # 3. Temporal consistency checker (Bi-LSTM)
        # Takes frame features over time (sequence)
        self.lstm = nn.LSTM(
            input_size=visual_features,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,  # Bi-directional means 512*2 = 1024 output size
            dropout=0.3
        )

        # 4. Attention mechanism for temporal features
        # Helps the model focus on the most important frames
        self.attention = nn.Sequential(
            nn.Linear(1024, 256),  # 1024 is the Bi-LSTM output size
            nn.Tanh(),
            nn.Linear(256, 1)  # Outputs a single score per frame
        )

        # 5. Fusion and classification head
        fusion_input = visual_features + 128 + 1024  # visual (avg) + audio (128) + temporal (1024)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),

            nn.Linear(256, num_classes)  # Final output (Real or Fake)
        )

    def forward(self, frames, audio_features=None):
        batch_size, num_frames, c, h, w = frames.shape

        # --- A. Visual and Temporal Processing ---

        # 1. Prepare for EfficientNet: Flatten batch and time dimensions
        frames_flat = frames.view(-1, c, h, w)
        visual_features = self.visual_backbone(frames_flat)
        # 2. Restore time dimension: [B*T, F] -> [B, T, F]
        visual_features = visual_features.view(batch_size, num_frames, -1)

        # 3. Temporal analysis with LSTM
        lstm_out, _ = self.lstm(visual_features)  # lstm_out shape: [B, T, 1024]

        # 4. Apply attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # weights shape: [B, T, 1]
        temporal_features = torch.sum(lstm_out * attention_weights, dim=1)  # shape: [B, 1024]

        # 5. Average visual features across all frames
        avg_visual = visual_features.mean(dim=1)  # shape: [B, visual_features]

        # --- B. Audio Processing ---
        if audio_features is not None:
            # audio_features shape is expected to be [B, 1, H, W] (Mel Spectrogram)
            audio_out = self.audio_conv(audio_features)
            audio_out = audio_out.view(batch_size, -1)  # shape: [B, 128]
        else:
            # If no audio, use zeros of the expected size (128 features)
            audio_out = torch.zeros(batch_size, 128).to(frames.device)

        # --- C. Fusion and Classification ---
        # Concatenate all three feature streams
        combined = torch.cat([avg_visual, audio_out, temporal_features], dim=1)
        output = self.fusion(combined)

        return output