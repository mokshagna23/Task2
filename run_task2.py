"""
Task 2: Complete Pipeline Runner
Executes full training, evaluation, and analysis workflow
"""

import subprocess
import sys
from pathlib import Path
import argparse
import json
from datetime import datetime


def print_section(title: str):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and report status
    
    Args:
        cmd: Command as list of arguments
        description: Description of what the command does
    
    Returns:
        bool: Success status
    """
    print(f"\n→ {description}...")
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {description} encountered error: {e}")
        return False


def main(args):
    """Execute complete Task 2 pipeline"""
    
    print_section("Task 2: RL Agent Training & Evaluation Pipeline")
    print("\nConfiguration:")
    print(f"  Episodes:           {args.episodes}")
    print(f"  Eval Episodes:      {args.eval_episodes}")
    print(f"  Train DQN:          {args.train_dqn or args.train_both}")
    print(f"  Train PPO:          {args.train_ppo or args.train_both}")
    print(f"  Seed (reproducibility): 42")
    print(f"  Focus: Sparse/Delayed Rewards")
    
    workspace_dir = Path(__file__).parent
    
    # Setup output directory
    print_section("Setting up output directories")
    
    import task2_setup
    print(f"✓ Output directories configured")
    
    # Install dependencies
    if args.install_deps:
        print_section("Installing dependencies")
        run_command(
            [sys.executable, "-m", "pip", "install", "-q", "-r", 
             str(workspace_dir / "task2_all_requirements.txt")],
            "Dependency installation"
        )
    
    # Run training
    print_section("Training RL Agents")
    
    train_cmd = [
        sys.executable,
        str(workspace_dir / "train_complete.py"),
        "--num-episodes", str(args.episodes),
        "--eval-episodes", str(args.eval_episodes)
    ]
    
    if args.train_dqn:
        train_cmd.append("--train-dqn")
    elif args.train_ppo:
        train_cmd.append("--train-ppo")
    else:
        train_cmd.append("--train-both")
    
    success = run_command(train_cmd, "Agent training")
    
    if not success and args.require_train:
        print("\n✗ Training failed. Cannot proceed with evaluation.")
        return False
    
    # Run evaluation
    if args.run_eval:
        print_section("Evaluating Trained Agents")
        
        eval_cmd = [
            sys.executable,
            str(workspace_dir / "evaluate_complete.py")
        ]
        
        if args.detailed_analysis:
            eval_cmd.append("--detailed")
        
        run_command(eval_cmd, "Agent evaluation and analysis")
    
    # Summary
    print_section("Pipeline Summary")
    
    from task2_setup import RESULTS_DIR, MODELS_DIR
    
    print(f"\nOutput locations:")
    print(f"  Models:  {MODELS_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    
    # List generated files
    if RESULTS_DIR.exists():
        print(f"\nGenerated files:")
        for f in sorted(RESULTS_DIR.glob("*")):
            print(f"  - {f.name}")
    
    print("\n" + "="*70)
    print("Pipeline completed!")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute complete Task 2 RL training and evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train both agents with default settings
  python run_task2.py
  
  # Train only DQN for 50 episodes
  python run_task2.py --train-dqn --episodes 50
  
  # Train PPO and skip evaluation
  python run_task2.py --train-ppo --no-eval
  
  # Full detailed analysis
  python run_task2.py --detailed-analysis --eval-episodes 50
        """
    )
    
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of training episodes (default: 100)')
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='Number of evaluation episodes (default: 20)')
    parser.add_argument('--train-dqn', action='store_true',
                       help='Train DQN agent only')
    parser.add_argument('--train-ppo', action='store_true',
                       help='Train PPO agent only')
    parser.add_argument('--train-both', action='store_true',
                       help='Train both agents (default if neither specified)')
    parser.add_argument('--no-eval', dest='run_eval', action='store_false',
                       help='Skip evaluation after training')
    parser.add_argument('--detailed-analysis', action='store_true',
                       help='Generate detailed analysis reports')
    parser.add_argument('--no-install-deps', dest='install_deps', action='store_false',
                       help='Skip dependency installation')
    parser.add_argument('--require-train', action='store_true',
                       help='Exit if training fails')
    
    # Set defaults
    parser.set_defaults(
        run_eval=True,
        install_deps=True,
        require_train=False
    )
    
    args = parser.parse_args()
    
    # Default to training both if neither specified
    if not (args.train_dqn or args.train_ppo):
        args.train_both = True
    
    try:
        success = main(args)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n✗ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Pipeline encountered unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
