"""
Task 2: Comprehensive Evaluation and Analysis Script
Evaluates, compares, and analyzes trained RL agents
"""

import numpy as np
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple

from task2_setup import MODELS_DIR, LOGS_DIR, RESULTS_DIR


class AgentAnalyzer:
    """Analyze and compare trained agents"""
    
    def __init__(self, results_dir: Path = RESULTS_DIR):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
    
    def load_results(self, filename: str = "training_summaries.json") -> Dict:
        """Load training results from JSON"""
        path = self.results_dir / filename
        if not path.exists():
            print(f"! Results file not found: {path}")
            return {}
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def plot_rewards(self, agents_dict: Dict):
        """
        Plot training rewards over episodes.
        
        Shows convergence and variance.
        """
        fig, axes = plt.subplots(1, len(agents_dict), figsize=(15, 4))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for idx, (agent_name, agent) in enumerate(agents_dict.items()):
            ax = axes[idx]
            rewards = agent.episode_rewards
            
            # Plot raw rewards
            ax.plot(rewards, alpha=0.3, label='Raw')
            
            # Plot moving average (100-episode window)
            if len(rewards) > 100:
                ma = np.convolve(rewards, np.ones(100)/100, mode='valid')
                ax.plot(range(99, len(rewards)), ma, label='MA-100', linewidth=2)
            
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.set_title(f'{agent_name}\nTotal Episodes: {len(rewards)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.results_dir / "rewards_training.png"
        plt.savefig(path, dpi=100)
        print(f"✓ Rewards plot saved: {path}")
        plt.close()
    
    def plot_losses(self, agents_dict: Dict):
        """
        Plot training losses over episodes.
        
        Shows learning stability.
        """
        fig, axes = plt.subplots(1, len(agents_dict), figsize=(15, 4))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        for idx, (agent_name, agent) in enumerate(agents_dict.items()):
            ax = axes[idx]
            losses = agent.episode_losses
            
            if len(losses) == 0:
                ax.text(0.5, 0.5, 'No loss data', ha='center', va='center')
                ax.set_title(f'{agent_name}\nNo loss data')
                continue
            
            # Plot raw losses
            ax.plot(losses, alpha=0.3, label='Raw')
            
            # Plot moving average
            if len(losses) > 50:
                ma = np.convolve(losses, np.ones(50)/50, mode='valid')
                ax.plot(range(49, len(losses)), ma, label='MA-50', linewidth=2)
            
            ax.set_xlabel('Update Step')
            ax.set_ylabel('Loss')
            ax.set_title(f'{agent_name}\nTotal Updates: {len(losses)}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = self.results_dir / "losses_training.png"
        plt.savefig(path, dpi=100)
        print(f"✓ Losses plot saved: {path}")
        plt.close()
    
    def generate_comparison_report(self, summaries: Dict) -> str:
        """
        Generate detailed comparison report of agents.
        """
        report = "\n" + "="*70 + "\n"
        report += "AGENT PERFORMANCE COMPARISON REPORT\n"
        report += "="*70 + "\n\n"
        
        # Summary table
        report += "Algorithm Performance Summary:\n"
        report += "-" * 70 + "\n"
        report += f"{'Agent':<20} {'Avg Reward':<15} {'Max Reward':<15} {'Min Reward':<15}\n"
        report += "-" * 70 + "\n"
        
        for agent_name, summary in summaries.items():
            avg = summary.get('avg_reward', 0.0)
            max_r = summary.get('max_reward', 0.0)
            min_r = summary.get('min_reward', 0.0)
            report += f"{agent_name:<20} {avg:<15.4f} {max_r:<15.4f} {min_r:<15.4f}\n"
        
        report += "\n" + "-" * 70 + "\n\n"
        
        # Detailed breakdown
        report += "Detailed Breakdown:\n"
        report += "-" * 70 + "\n"
        
        for agent_name, summary in summaries.items():
            report += f"\n{agent_name}:\n"
            for key, value in summary.items():
                if key not in ['agent_name', 'timestamp']:
                    report += f"  {key:<30} {value}\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
    
    def generate_analysis_report(self, agents_dict: Dict) -> str:
        """
        Generate detailed analysis of agent behaviors and learning.
        """
        report = "\n" + "="*70 + "\n"
        report += "DETAILED AGENT ANALYSIS\n"
        report += "="*70 + "\n\n"
        
        for agent_name, agent in agents_dict.items():
            report += f"\n{agent_name} Analysis\n"
            report += "-" * 70 + "\n"
            
            if agent.episode_rewards:
                rewards = np.array(agent.episode_rewards)
                report += f"Rewards:\n"
                report += f"  Mean:     {np.mean(rewards):.4f}\n"
                report += f"  Std Dev:  {np.std(rewards):.4f}\n"
                report += f"  Max:      {np.max(rewards):.4f}\n"
                report += f"  Min:      {np.min(rewards):.4f}\n"
                report += f"  Median:   {np.median(rewards):.4f}\n"
                
                # Trend analysis (first half vs last quarter)
                if len(rewards) >= 4:
                    first_half = np.mean(rewards[:len(rewards)//2])
                    last_quarter = np.mean(rewards[3*len(rewards)//4:])
                    improvement = last_quarter - first_half
                    report += f"\n  Early performance (avg): {first_half:.4f}\n"
                    report += f"  Late performance (avg):  {last_quarter:.4f}\n"
                    report += f"  Improvement:             {improvement:.4f}\n"
            
            if agent.episode_losses:
                losses = np.array(agent.episode_losses)
                report += f"\nLosses:\n"
                report += f"  Mean:     {np.mean(losses):.6f}\n"
                report += f"  Std Dev:  {np.std(losses):.6f}\n"
                report += f"  Max:      {np.max(losses):.6f}\n"
                report += f"  Min:      {np.min(losses):.6f}\n"
            
            report += f"\nTraining Info:\n"
            report += f"  Total Episodes: {len(agent.episode_rewards)}\n"
            report += f"  Total Steps:    {agent.total_steps}\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report
    
    def analyze_sparse_rewards(self, agents_dict: Dict) -> str:
        """
        Analyze how well agents handle sparse/delayed rewards.
        """
        report = "\n" + "="*70 + "\n"
        report += "SPARSE REWARD LEARNING ANALYSIS\n"
        report += "="*70 + "\n\n"
        
        report += "Assessment Criteria:\n"
        report += "- Convergence speed (episodes to stable learning)\n"
        report += "- Reward variance (stability with sparse signals)\n"
        report += "- Learning efficiency (reward per step)\n"
        report += "- Recovery from poor starts (variance in early episodes)\n\n"
        
        for agent_name, agent in agents_dict.items():
            report += f"\n{agent_name} - Sparse Reward Handling:\n"
            report += "-" * 70 + "\n"
            
            rewards = agent.episode_rewards
            if len(rewards) < 10:
                report += "  Insufficient data for analysis\n"
                continue
            
            # Convergence: measure coefficient of variation over time
            first_third = rewards[:len(rewards)//3]
            last_third = rewards[2*len(rewards)//3:]
            
            early_cv = np.std(first_third) / (abs(np.mean(first_third)) + 1e-8)
            late_cv = np.std(last_third) / (abs(np.mean(last_third)) + 1e-8)
            
            report += f"  Early episodes (CV):      {early_cv:.4f}\n"
            report += f"  Late episodes (CV):       {late_cv:.4f}\n"
            report += f"  Stability improvement:    {early_cv - late_cv:.4f}\n"
            
            # Learning efficiency
            avg_reward = np.mean(rewards)
            efficiency = avg_reward / (agent.total_steps / len(rewards) + 1e-8)
            report += f"\n  Average reward/step:      {efficiency:.6f}\n"
            
            # Episodes to stable learning (when CV < 0.5)
            cv_history = []
            window = max(10, len(rewards)//10)
            for i in range(window, len(rewards)):
                window_rewards = rewards[i-window:i]
                cv = np.std(window_rewards) / (abs(np.mean(window_rewards)) + 1e-8)
                cv_history.append(cv)
            
            if cv_history:
                stable_threshold = 0.5
                stable_episodes = next((i for i, cv in enumerate(cv_history) if cv < stable_threshold),
                                      len(cv_history))
                report += f"  Episodes to stability:    {stable_episodes + window}\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report


def main(args):
    """Main analysis pipeline"""
    
    print("\n" + "="*70)
    print("Task 2: Agent Analysis and Evaluation")
    print("="*70)
    
    analyzer = AgentAnalyzer()
    
    # Load results
    print("\n→ Loading training results...")
    summaries = analyzer.load_results()
    
    if not summaries:
        print("! No training results found. Please run training first.")
        return
    
    print(f"✓ Loaded summaries for {len(summaries)} agents")
    
    # Try to load agents for detailed analysis
    agents_dict = {}
    print("\n→ Loading trained agent models...")
    
    try:
        from task2_dqn_agent_v2 import DQNAgent
        from task2_ppo_agent_v2 import PPOAgent
        
        # Note: This is simplified - actual loading would need to restore model weights
        # For now, we'll work with summaries only
        print("  (Agent models with full history would require model serialization)")
    except Exception as e:
        print(f"  Note: {e}")
    
    # Generate reports
    print("\n→ Generating analysis reports...")
    
    # Comparison report
    comparison_report = analyzer.generate_comparison_report(summaries)
    print(comparison_report)
    
    # Save comparison report
    report_path = RESULTS_DIR / "comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write(comparison_report)
    print(f"✓ Comparison report saved: {report_path}")
    
    # Sparse reward analysis
    sparse_report = analyzer.analyze_sparse_rewards(agents_dict) if agents_dict else ""
    if sparse_report:
        report_path = RESULTS_DIR / "sparse_reward_analysis.txt"
        with open(report_path, 'w') as f:
            f.write(sparse_report)
        print(f"✓ Sparse reward analysis saved: {report_path}")
    
    # Generate visualizations if we have agent data
    if agents_dict:
        print("\n→ Generating visualizations...")
        try:
            analyzer.plot_rewards(agents_dict)
            analyzer.plot_losses(agents_dict)
        except Exception as e:
            print(f"! Error generating plots: {e}")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print(f"Results saved in: {RESULTS_DIR}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze trained RL agents")
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed analysis reports')
    args = parser.parse_args()
    
    main(args)
