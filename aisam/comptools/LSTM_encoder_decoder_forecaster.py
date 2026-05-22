import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from aisam.comptools.regression_forecaster import regression_metrics


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()
        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.0):
        super().__init__()
        lstm_dropout = dropout if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        outputs, _ = self.lstm(x, (hidden, cell))
        return self.fc(outputs)


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder_input_dim, decoder_input_dim, hidden_dim, n_layers, dropout=0.0):
        super().__init__()
        self.encoder = Encoder(encoder_input_dim, hidden_dim, n_layers, dropout=dropout)
        self.decoder = Decoder(decoder_input_dim, hidden_dim, 1, n_layers, dropout=dropout)

    def forward(self, past_xu, future_u):
        hidden, cell = self.encoder(past_xu)
        return self.decoder(future_u, hidden, cell)


class LSTMEncoderDecoderForecaster:
    """
    Encoder-decoder LSTM forecaster compatible with AISAM window datasets.

    The model consumes the same flattened X matrices produced by
    `make_window_dataset`: past features, past inputs, and future inputs.
    """

    def __init__(
        self,
        past_feature_window,
        future_window=1,
        past_input_window=None,
        future_input_window=None,
        feature_dim=1,
        input_dim=1,
        hidden_size=64,
        num_layers=2,
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
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
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
        self.decoder_mean_ = None
        self.decoder_std_ = None
        self.y_mean_ = None
        self.y_std_ = None
        self.training_history_ = []

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(int(self.random_state))
            np.random.seed(int(self.random_state))

        past_xu, future_u = self._split_x(X)
        y_seq = np.asarray(y, dtype=float).reshape(-1, self.future_window, 1)
        if self.normalize:
            past_xu, future_u, y_seq = self._fit_normalizers(past_xu, future_u, y_seq)

        dataset = TensorDataset(
            torch.tensor(past_xu, dtype=torch.float32),
            torch.tensor(future_u, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = Seq2SeqLSTM(
            encoder_input_dim=self.feature_dim + self.input_dim,
            decoder_input_dim=self.input_dim,
            hidden_dim=self.hidden_size,
            n_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.training_history_ = []
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_past, batch_future_u, batch_y in loader:
                batch_past = batch_past.to(self.device)
                batch_future_u = batch_future_u.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.model(batch_past, batch_future_u)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
            mean_loss = total_loss / max(1, len(loader))
            self.training_history_.append(mean_loss)
            if self.verbose:
                print(f"LSTM epoch {epoch + 1}/{self.epochs}, loss={mean_loss:.6f}")
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model has not been fit yet.")
        past_xu, future_u = self._split_x(X)
        if self.normalize:
            past_xu = (past_xu - self.encoder_mean_) / self.encoder_std_
            future_u = (future_u - self.decoder_mean_) / self.decoder_std_
        self.model.eval()
        predictions = []
        loader = DataLoader(
            TensorDataset(
                torch.tensor(past_xu, dtype=torch.float32),
                torch.tensor(future_u, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            for batch_past, batch_future_u in loader:
                output = self.model(batch_past.to(self.device), batch_future_u.to(self.device))
                predictions.append(output.cpu().numpy())
        pred = np.concatenate(predictions, axis=0)
        if self.normalize:
            pred = pred * self.y_std_ + self.y_mean_
        return pred.reshape(-1, self.future_window)

    def evaluate(self, X, y):
        return regression_metrics(y, self.predict(X))

    def _split_x(self, X):
        if self.past_input_window != self.past_feature_window:
            raise ValueError("LSTM forecaster requires past_input_window == past_feature_window.")
        if self.future_input_window < self.future_window:
            raise ValueError("LSTM forecaster requires future_input_window >= future_window.")

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
        future_u = X[:, future_start:future_start + future_input_size].reshape(
            n,
            self.future_input_window,
            self.input_dim,
        )[:, :self.future_window, :]
        past_xu = np.concatenate([feature_hist, input_hist], axis=2)
        return past_xu, future_u

    def _fit_normalizers(self, past_xu, future_u, y_seq):
        self.encoder_mean_ = past_xu.mean(axis=(0, 1), keepdims=True)
        self.encoder_std_ = past_xu.std(axis=(0, 1), keepdims=True)
        self.encoder_std_[self.encoder_std_ == 0] = 1.0

        self.decoder_mean_ = future_u.mean(axis=(0, 1), keepdims=True)
        self.decoder_std_ = future_u.std(axis=(0, 1), keepdims=True)
        self.decoder_std_[self.decoder_std_ == 0] = 1.0

        self.y_mean_ = y_seq.mean(axis=(0, 1), keepdims=True)
        self.y_std_ = y_seq.std(axis=(0, 1), keepdims=True)
        self.y_std_[self.y_std_ == 0] = 1.0

        return (
            (past_xu - self.encoder_mean_) / self.encoder_std_,
            (future_u - self.decoder_mean_) / self.decoder_std_,
            (y_seq - self.y_mean_) / self.y_std_,
        )


def train_lstm_forecaster(
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
    forecaster = LSTMEncoderDecoderForecaster(
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
