import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class Plotter:
    def __init__(self, data_wavelength, data_reflectance, R_cal, params, simulated_params=None, optional_data={}):
        """
        Initialize the Plotter with data and results.
        
        Args:
            data_wavelength (np.array): Wavelength data.
            data_reflectance (np.array): Reflectance data.
            R_cal (np.array): Calculated reflectance.
            params (np.array): Predicted parameters.
            simulated_params (np.array, optional): True parameters (if simulated). Defaults to None.
            optional_data (dict, optional): Additional data for plotting (e.g., n, k, optimized results). Defaults to {}.
        """
        self.data_wavelength = data_wavelength
        self.data_reflectance = data_reflectance
        self.R_cal = R_cal
        self.params = params
        self.simulated_params = simulated_params
        self.optional_data = optional_data
        self.energy = 1240 / data_wavelength  # Convert wavelength to energy

    def plot_reflectance(self, ax, label=""):
        """Plot reflectance values (data, predicted, and optimized if available)."""
        ax.plot(self.energy, self.data_reflectance, label="R_data", alpha=0.5)
        ax.plot(self.energy, self.R_cal, label="R_pred", color="orange", linestyle="dashed")
        if 'R_cal_opt' in self.optional_data and self.optional_data['R_cal_opt'] is not None:
            ax.plot(self.energy, self.optional_data['R_cal_opt'], label="R_fit", color="r")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("Reflectance")
        ax.set_title("Reflectance", fontweight='bold')
        ax.legend()
        self._add_subplot_label(ax, label)

    def plot_n(self, ax, label=""):
        """Plot n values (true, predicted, and optimized if available)."""
        if 'n_true' in self.optional_data and self.optional_data['n_true'] is not None:
            ax.plot(self.energy, self.optional_data['n_true'], label="n_true", alpha=0.5)
        if 'n' in self.optional_data and self.optional_data['n'] is not None:
            ax.plot(self.energy, self.optional_data['n'], label="n_pred", color="orange", linestyle="dashed")
            
        if 'n_true' in self.optional_data and 'n' in self.optional_data and self.optional_data['n_true'] is not None and self.optional_data['n'] is not None:
            diff = np.abs(self.optional_data['n_true'] - self.optional_data['n'])
            max_diff_index = np.argmax(diff)
            max_diff_energy = self.energy[max_diff_index]
            max_diff = diff[max_diff_index]
            percentage = max_diff / max(np.max(self.optional_data['n_true']), np.max(self.optional_data['n'])) * 100
            ax.annotate("", xy=(max_diff_energy, self.optional_data['n_true'][max_diff_index]), xytext=(max_diff_energy, self.optional_data['n'][max_diff_index]), arrowprops=dict(arrowstyle="<->"))
            ax.text(max_diff_energy, (self.optional_data['n_true'][max_diff_index] + self.optional_data['n'][max_diff_index]) / 2, f'Δ = {percentage:.1f}%', fontsize='14', fontweight='bold', ha='right')
            
        if 'n_opt' in self.optional_data and self.optional_data['n_opt'] is not None:
            ax.plot(self.energy, self.optional_data['n_opt'], label="n_fit", color="r")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("n")
        ax.set_title("n", fontweight='bold')
        ax.legend()
        self._add_subplot_label(ax, label)

    def plot_k(self, ax, label=""):
        """Plot k values (true, predicted, and optimized if available)."""
        if 'k_true' in self.optional_data and self.optional_data['k_true'] is not None:
            ax.plot(self.energy, self.optional_data['k_true'], label="k_true", alpha=0.5)
        if 'k' in self.optional_data and self.optional_data['k'] is not None:
            ax.plot(self.energy, self.optional_data['k'], label="k_pred", color="orange", linestyle="dashed")
            
        if 'k_true' in self.optional_data and 'k' in self.optional_data and self.optional_data['k_true'] is not None and self.optional_data['k'] is not None:
            diff = np.abs(self.optional_data['k_true'] - self.optional_data['k'])
            max_diff_index = np.argmax(diff)
            max_diff_energy = self.energy[max_diff_index]
            max_diff = diff[max_diff_index]
            percentage = max_diff / max(np.max(self.optional_data['k_true']), np.max(self.optional_data['k'])) * 100
            ax.annotate("", xy=(max_diff_energy, self.optional_data['k_true'][max_diff_index]), xytext=(max_diff_energy, self.optional_data['k'][max_diff_index]), arrowprops=dict(arrowstyle="<->"))
            ax.text(max_diff_energy, (self.optional_data['k_true'][max_diff_index] + self.optional_data['k'][max_diff_index]) / 2, f'Δ = {percentage:.1f}%', fontsize='14', fontweight='bold', ha='right')
            
        if 'k_opt' in self.optional_data and self.optional_data['k_opt'] is not None:
            ax.plot(self.energy, self.optional_data['k_opt'], label="k_fit", color="r")
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("k")
        ax.set_title("k", fontweight='bold')
        ax.legend()
        self._add_subplot_label(ax, label)

    def plot_radar_chart(self, ax, label=""):
        """Plot a radar chart comparing true and predicted parameters."""
        labels = ['A', 'E0', 'G', 'Eg', 'e_inf', 'thickness']
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]  # Close the circle

        if self.simulated_params is not None:
            true_params = np.concatenate((self.simulated_params, [self.simulated_params[0]]))
            pred_params = np.concatenate((self.params, [self.params[0]]))
            true_params_normalized = np.full_like(true_params, 0.5)
            pred_params_normalized = pred_params / true_params * 0.5

            true_polygon = Polygon(list(zip(angles, true_params_normalized)), closed=True, alpha=0.25)
            pred_polygon = Polygon(list(zip(angles, pred_params_normalized)), closed=True, color='orange', alpha=0.25)

            ax.add_patch(true_polygon)
            ax.plot(angles, true_params_normalized, linewidth=1, label='True')
            ax.add_patch(pred_polygon)
            ax.plot(angles, pred_params_normalized, linewidth=1, label='Pred')

            for angle, label_param, tp, pp in zip(angles, labels, true_params, pred_params):
                ax.text(angle, 1.2, f'{tp:.2f}', ha='center', va='center', fontweight='bold', color='#1f77b4', fontsize=6)
                ax.text(angle, (pp / tp * 0.5) + 2.0, f'{pp:.2f}', ha='center', va='center', color='orange', fontweight='bold', fontsize=6)

        ax.set_title('Parameters Comparison', fontweight='bold')
        ax.grid(alpha=0.2)
        ax.set_yticklabels([])
        ax.legend(loc='upper left', bbox_to_anchor=(-0.4, 1))
        self._add_subplot_label(ax, label, x=-0.43, y=1.15)

    def plot_single_results(self, suptitle, save=True, show=False, save_dir="results/plots"):
        """Plot all results in a single figure for a single data point."""
        fig = plt.figure(figsize=(10, 8))
        plt.suptitle(suptitle, fontsize=16)
        grid_spec = plt.GridSpec(2, 2)

        ax_radar = fig.add_subplot(grid_spec[0, 0], polar=True)
        ax_n = fig.add_subplot(grid_spec[0, 1])
        ax_k = fig.add_subplot(grid_spec[1, 0])
        ax_reflectance = fig.add_subplot(grid_spec[1, 1])

        # Plot each component
        self.plot_radar_chart(ax_radar, label="(a)")
        self.plot_n(ax_n, label="(b)")
        self.plot_k(ax_k, label="(c)")
        self.plot_reflectance(ax_reflectance, label="(d)")

        plt.tight_layout()
        if save:
            plt.savefig(f"{save_dir}/{suptitle}.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plot_dataset_results(self, plot_suptitle, save=False, show=True):
        """Plot results for the entire dataset."""
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        plt.suptitle(plot_suptitle, fontsize=16)

        # Plot reflectance
        for i in range(self.data_reflectance.shape[0]):
            axes[0, 0].plot(self.energy, self.data_reflectance[i, :], alpha=0.1, color='gray')
        axes[0, 0].plot(self.energy, np.mean(self.data_reflectance, axis=0), label="Mean R_data", color='blue')
        axes[0, 0].set_xlabel("Energy (eV)")
        axes[0, 0].set_ylabel("Reflectance")
        axes[0, 0].set_title("Reflectance", fontweight='bold')
        axes[0, 0].legend()

        # Plot n
        if 'n' in self.optional_data and self.optional_data['n'] is not None:
            for i in range(self.optional_data['n'].shape[0]):
                axes[0, 1].plot(self.energy, self.optional_data['n'][i, :], alpha=0.1, color='gray')
            axes[0, 1].plot(self.energy, np.mean(self.optional_data['n'], axis=0), label="Mean n_pred", color='orange')
            axes[0, 1].set_xlabel("Energy (eV)")
            axes[0, 1].set_ylabel("n")
            axes[0, 1].set_title("n", fontweight='bold')
            axes[0, 1].legend()

        # Plot k
        if 'k' in self.optional_data and self.optional_data['k'] is not None:
            for i in range(self.optional_data['k'].shape[0]):
                axes[1, 0].plot(self.energy, self.optional_data['k'][i, :], alpha=0.1, color='gray')
            axes[1, 0].plot(self.energy, np.mean(self.optional_data['k'], axis=0), label="Mean k_pred", color='orange')
            axes[1, 0].set_xlabel("Energy (eV)")
            axes[1, 0].set_ylabel("k")
            axes[1, 0].set_title("k", fontweight='bold')
            axes[1, 0].legend()

        # Plot parameters
        if 'params' in self.optional_data and self.optional_data['params'] is not None:
            param_names = ['A', 'E0', 'G', 'Eg', 'e_inf', 'thickness']
            param_values = np.array(self.optional_data['params'])
            for i, name in enumerate(param_names):
                axes[1, 1].scatter([i] * param_values.shape[0], param_values[:, i], alpha=0.5, label=name)
            axes[1, 1].set_xticks(range(len(param_names)))
            axes[1, 1].set_xticklabels(param_names)
            axes[1, 1].set_ylabel("Parameter Value")
            axes[1, 1].set_title("Parameters", fontweight='bold')
            axes[1, 1].legend()

        plt.tight_layout()
        if save:
            plt.savefig(f"{plot_suptitle}_dataset.png", dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def _add_subplot_label(self, ax, label, x=-0.1, y=1.15):
        """Add a label to a subplot."""
        ax.text(x, y, label, transform=ax.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')