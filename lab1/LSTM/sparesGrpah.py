import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D

# Load Optuna study from memory (assuming you've just run it)
# If you're saving to disk instead, use joblib or optuna.load_study

def visualize_3d_surface(study, x_param, y_param, metric_name="Accuracy"):
    # Extract relevant trial data
    trials_df = pd.DataFrame([
        {
            x_param: t.params.get(x_param),
            y_param: t.params.get(y_param),
            'value': t.value
        }
        for t in study.trials
        if t.value is not None and x_param in t.params and y_param in t.params
    ])

    if trials_df.empty:
        print(f"No valid data for parameters: {x_param}, {y_param}")
        return

    # Create grid
    x = trials_df[x_param]
    y = trials_df[y_param]
    z = trials_df['value']

    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='linear')

    # Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap='plasma', edgecolor='k', alpha=0.9)

    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_zlabel(metric_name)
    ax.set_title(f'Hyperparameters Performance Landscape ({x_param} & {y_param})')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Re-run your study in memory or load it here
    from hyperparameter_tuning import objective  # Import your original objective
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=500)

    # Visualize
    visualize_3d_surface(study, x_param="hidden_dim", y_param="batch_size")
