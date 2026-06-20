import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from aisam.comptools.regression_forecaster import regression_metrics


class LSTMMLPNetwork(nn.Module):
    """
    LSTM encoder plus MLP decoder.

    This mirrors the legacy Keras LSTM-MLP shape:
    encoder_0 returns a sequence, encoder_1 returns final state_h/state_c,
    and the decoder predicts the output horizon from [state_h, state_c,
    future_light]. `future_light` is the flattened future input plan, whose
    length is controlled by future_input_window rather than future_window.
    """

    def __init__(
        self,
        encoder_input_dim,
        future_input_dim,
        future_window,
        lstm_units=64,
        latent_dim=64,
        mlp_layers=2,
        mlp_dim=64,
        dropout=0.0,
    ):
        super().__init__()
        self.encoder_0 = nn.LSTM(
            encoder_input_dim,
            lstm_units,
            batch_first=True,
        )
        self.encoder_1 = nn.LSTM(
            lstm_units,
            latent_dim,
            batch_first=True,
        )

        decoder_layers = []
        decoder_input_dim = 2 * latent_dim + future_input_dim
        for layer_index in range(int(mlp_layers)):
            in_features = decoder_input_dim if layer_index == 0 else mlp_dim
            decoder_layers.append(nn.Linear(in_features, mlp_dim))
            decoder_layers.append(nn.ReLU())
            if dropout:
                decoder_layers.append(nn.Dropout(dropout))
        output_input_dim = mlp_dim if int(mlp_layers) > 0 else decoder_input_dim
        decoder_layers.append(nn.Linear(output_input_dim, future_window))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, past_xu, future_light):
        encoder_sequence, _ = self.encoder_0(past_xu)
        _, (hidden, cell) = self.encoder_1(encoder_sequence)
        state_h = hidden[-1]
        state_c = cell[-1]
        decoder_input = torch.cat([state_h, state_c, future_light], dim=1)
        return self.decoder(decoder_input)


class LSTMMLPForecaster:
    """
    Torch LSTM-MLP forecaster compatible with AISAM window datasets.

    The AISAM window matrix is split into past features, past stimulation, and
    future stimulation. Past gene-expression features and past stimulation are
    encoded together by two LSTM layers; the final hidden and cell states are
    concatenated with the full future stimulation plan and decoded by an MLP
    into the prediction horizon.
    """

    def __init__(
        self,
        past_feature_window,
        future_window=1,
        past_input_window=None,
        future_input_window=None,
        feature_dim=1,
        input_dim=1,
        lstm_units=64,
        latent_dim=64,
        mlp_layers=2,
        mlp_dim=64,
        dropout=0.0,
        batch_size=32,
        epochs=10,
        learning_rate=0.001,
        normalize=True,
        device=None,
        random_state=None,
        verbose=True,
    ):
        self.past_feature_window = int(past_feature_window)
        self.future_window = int(future_window)
        self.past_input_window = int(past_input_window or past_feature_window)
        self.future_input_window = int(future_input_window or future_window)
        self.feature_dim = int(feature_dim)
        self.input_dim = int(input_dim)
        self.lstm_units = int(lstm_units)
        self.latent_dim = int(latent_dim)
        self.mlp_layers = int(mlp_layers)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.normalize = bool(normalize)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.verbose = bool(verbose)
        self.model = None
        self.encoder_mean_ = None
        self.encoder_std_ = None
        self.future_mean_ = None
        self.future_std_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self.training_history_ = []

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
            np.random.seed(int(self.random_state))

        past_xu, future_light = self._split_x(X)
        y = np.asarray(y, dtype=float).reshape(-1, self.future_window)
        if self.normalize:
            past_xu, future_light, y = self._fit_normalizers(past_xu, future_light, y)

        dataset = TensorDataset(
            torch.tensor(past_xu, dtype=torch.float32),
            torch.tensor(future_light, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = LSTMMLPNetwork(
            encoder_input_dim=self.feature_dim + self.input_dim,
            future_input_dim=self.future_input_window * self.input_dim,
            future_window=self.future_window,
            lstm_units=self.lstm_units,
            latent_dim=self.latent_dim,
            mlp_layers=self.mlp_layers,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.training_history_ = []
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_past, batch_future_light, batch_y in loader:
                batch_past = batch_past.to(self.device)
                batch_future_light = batch_future_light.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_past, batch_future_light)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            mean_loss = total_loss / max(1, len(loader))
            self.training_history_.append(mean_loss)
            if self.verbose:
                print(f"LSTM-MLP epoch {epoch + 1}/{self.epochs}, loss={mean_loss:.6f}")
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fit yet.")
        past_xu, future_light = self._split_x(X)
        if self.normalize:
            past_xu = (past_xu - self.encoder_mean_) / self.encoder_std_
            future_light = (future_light - self.future_mean_) / self.future_std_

        self.model.eval()
        predictions = []
        loader = DataLoader(
            TensorDataset(
                torch.tensor(past_xu, dtype=torch.float32),
                torch.tensor(future_light, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            for batch_past, batch_future_light in loader:
                output = self.model(
                    batch_past.to(self.device),
                    batch_future_light.to(self.device),
                )
                predictions.append(output.cpu().numpy())
        pred = np.concatenate(predictions, axis=0)
        if self.normalize:
            pred = pred * self.y_std_ + self.y_mean_
        return pred.reshape(-1, self.future_window)

    def evaluate(self, X, y):
        return regression_metrics(y, self.predict(X))

    def _split_x(self, X):
        if self.past_input_window != self.past_feature_window:
            raise ValueError("LSTM-MLP forecaster requires past_input_window == past_feature_window.")

        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        feature_size = self.past_feature_window * self.feature_dim
        input_size = self.past_input_window * self.input_dim
        future_input_size = self.future_input_window * self.input_dim
        expected = feature_size + input_size + future_input_size
        if X.shape[1] != expected:
            raise ValueError(f"Expected X with {expected} columns, got {X.shape[1]}.")

        feature_hist = X[:, :feature_size].reshape(n, self.past_feature_window, self.feature_dim)
        input_start = feature_size
        input_hist = X[:, input_start:input_start + input_size].reshape(
            n,
            self.past_input_window,
            self.input_dim,
        )
        future_start = input_start + input_size
        future_inputs = X[:, future_start:future_start + future_input_size].reshape(
            n,
            self.future_input_window,
            self.input_dim,
        )
        future_light = future_inputs.reshape(n, -1)
        past_xu = np.concatenate([feature_hist, input_hist], axis=2)
        return past_xu, future_light

    def _fit_normalizers(self, past_xu, future_light, y):
        self.encoder_mean_ = past_xu.mean(axis=(0, 1), keepdims=True)
        self.encoder_std_ = past_xu.std(axis=(0, 1), keepdims=True)
        self.encoder_std_[self.encoder_std_ == 0] = 1.0

        self.future_mean_ = future_light.mean(axis=0, keepdims=True)
        self.future_std_ = future_light.std(axis=0, keepdims=True)
        self.future_std_[self.future_std_ == 0] = 1.0

        self.y_mean_ = y.mean(axis=0, keepdims=True)
        self.y_std_ = y.std(axis=0, keepdims=True)
        self.y_std_[self.y_std_ == 0] = 1.0

        return (
            (past_xu - self.encoder_mean_) / self.encoder_std_,
            (future_light - self.future_mean_) / self.future_std_,
            (y - self.y_mean_) / self.y_std_,
        )


def train_lstm_mlp_forecaster(
    X_train,
    y_train,
    past_feature_window,
    future_window=1,
    past_input_window=None,
    future_input_window=None,
    feature_dim=1,
    input_dim=1,
    **kwargs,
):
    forecaster = LSTMMLPForecaster(
        past_feature_window=past_feature_window,
        future_window=future_window,
        past_input_window=past_input_window,
        future_input_window=future_input_window,
        feature_dim=feature_dim,
        input_dim=input_dim,
        **kwargs,
    )
    forecaster.fit(X_train, y_train)
    return forecaster
