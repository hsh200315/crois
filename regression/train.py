import matplotlib.pyplot as plt
import joblib
import torch
import matplotlib.pyplot as plt
import torch
import joblib
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
import numpy as np
import mlflow


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=20,
):
    model.to(device)

    # 학습 데이터로 preprocessor 학습
    all_train_features = []
    for features, _ in train_loader:
        all_train_features.append(features.numpy())
    all_train_features = np.concatenate(all_train_features, axis=0)
    model.preprocessor.fit_transform(all_train_features)

    # Track losses
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.numpy(), labels.numpy()

            # 전처리 수행
            scaled_features = torch.FloatTensor(
                model.preprocessor.transform(features)
            ).to(device)

            labels = torch.FloatTensor(labels).reshape(-1, 1).to(device)

            # Reshape features to (batch_size, sequence_length, input_size)
            scaled_features = scaled_features.unsqueeze(
                1
            )  # Adding a sequence dimension

            optimizer.zero_grad()
            outputs = model(scaled_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.numpy(), labels.numpy()

                # 전처리 수행
                scaled_features = torch.FloatTensor(
                    model.preprocessor.transform(features)
                ).to(device)

                # Reshape features to (batch_size, sequence_length, input_size)
                scaled_features = scaled_features.unsqueeze(
                    1
                )  # Adding a sequence dimension

                labels = torch.FloatTensor(labels).reshape(-1, 1).to(device)

                outputs = model(scaled_features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # Record losses
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

        mlflow.log_metric("train_loss", train_losses[-1], step=epoch)
        mlflow.log_metric("val_loss", val_losses[-1], step=epoch)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}"
        )

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_plot.png")
    mlflow.log_artifact("loss_plot.png")
    plt.close()

    return train_losses, val_losses


def evaluate_model(
    model, test_loader, device, scaler_path="scaler.pkl", visualize=True
):
    model.eval()

    # Load the saved scaler
    scaler = joblib.load(scaler_path)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.numpy()
            labels = labels.numpy()

            # Transform test features using the fitted scaler
            scaled_features = torch.tensor(
                scaler.transform(features), dtype=torch.float32
            ).to(device)
            labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

            # Reshape features to (batch_size, sequence_length, input_size)
            scaled_features = scaled_features.unsqueeze(
                1
            )  # Adding a sequence dimension

            outputs = model(scaled_features)
            preds = outputs.cpu().numpy()
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy().flatten())

    # Calculate regression metrics
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    mae = mean_absolute_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    mape = (
        np.mean(
            np.abs((np.array(all_labels) - np.array(all_preds)) / np.array(all_labels))
        )
        * 100
    )
    explained_variance = explained_variance_score(all_labels, all_preds)

    # Print metrics
    print(f"Evaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Explained Variance Score: {explained_variance:.4f}")

    # Visualization in time order
    if visualize:
        plt.figure(figsize=(12, 6))
        time = range(len(all_labels))  # Time axis based on the sequence index

        # Plot true values
        plt.plot(
            time,
            all_labels,
            label="True Labels",
            color="green",
            alpha=0.7,
            linestyle="-",
            marker="o",
        )

        # Plot predicted values
        plt.plot(
            time,
            all_preds,
            label="Predicted Labels",
            color="blue",
            alpha=0.7,
            linestyle="-",
            marker="x",
        )

        # Trendlines (optional, for smoothing effect)
        from scipy.signal import savgol_filter

        true_trend = savgol_filter(all_labels, window_length=15, polyorder=2)
        pred_trend = savgol_filter(all_preds, window_length=15, polyorder=2)
        plt.plot(
            time,
            true_trend,
            label="True Trend",
            color="green",
            linestyle="--",
            linewidth=1.5,
        )
        plt.plot(
            time,
            pred_trend,
            label="Predicted Trend",
            color="blue",
            linestyle="--",
            linewidth=1.5,
        )

        # Labels, title, and legend
        plt.xlabel("Time (Sample Index)")
        plt.ylabel("Value")
        plt.title("True vs Predicted Labels Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    return (
        {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "explained_variance_score": explained_variance,
        },
        all_preds,
        all_labels,
    )


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
import numpy as np
import matplotlib.pyplot as plt


def evaluate_model_ML(model, X_test, y_test, visualize=True):
    # 예측 수행
    y_pred = model.predict(X_test)

    # 평가 지표 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    explained_variance = explained_variance_score(y_test, y_pred)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "explained_variance_score": explained_variance,
    }

    # 결과 시각화
    if visualize:
        plt.figure(figsize=(12, 6))
        time = range(len(y_test))  # Time axis

        # Plot true vs predicted values
        plt.plot(
            time,
            y_test,
            label="True Labels",
            color="green",
            alpha=0.7,
            linestyle="-",
            marker="o",
        )
        plt.plot(
            time,
            y_pred,
            label="Predicted Labels",
            color="blue",
            alpha=0.7,
            linestyle="-",
            marker="x",
        )

        # Labels, title, and legend
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title("True vs Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.show()

    return metrics, y_pred, y_test


def evaluate_model_ML_s(model, X_test, y_test, visualize=True):
    # Scikit-learn 기반 모델의 예측
    y_pred = model.predict(X_test)

    # 평가 지표 계산
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    explained_variance = explained_variance_score(y_test, y_pred)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "explained_variance_score": explained_variance,
    }

    # 결과 시각화
    if visualize:
        plt.figure(figsize=(12, 6))
        time = range(len(y_test))  # Time axis

        # Plot true vs predicted values
        plt.plot(
            time,
            y_test,
            label="True Labels",
            color="green",
            alpha=0.7,
            linestyle="-",
            marker="o",
        )
        plt.plot(
            time,
            y_pred,
            label="Predicted Labels",
            color="blue",
            alpha=0.7,
            linestyle="-",
            marker="x",
        )

        # Labels, title, and legend
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title("True vs Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.show()

    return metrics, y_pred, y_test
