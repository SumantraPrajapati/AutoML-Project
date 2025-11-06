import typer
from core import data_loader, preprocessor, feature_engineering, model_selector, trainer, evaluator

app = typer.Typer(help="AutoML-CLI — A Command Line Machine Learning System")

@app.command()
def train(
    data: str = typer.Option(..., "--data", help="Path to dataset (CSV/XLSX)"),
    target: str = typer.Option(..., "--target", help="Target column name")
):
    """Train model automatically"""
    df = data_loader.load_data(data , target)
    df = preprocessor.clean_data(df)
    df = feature_engineering.create_features(df)

    tr = trainer.Train_Model(problem_type='regression' , tuning_enabled=True)

    model, metrics = tr.train_auto(df, target)
    typer.echo(f"Model trained successfully! Accuracy: {metrics.get('accuracy', 'N/A')}")

@app.command()
def evaluate(model_path: str, test_data: str):
    """Evaluate a saved model"""
    evaluator.evaluate_model(model_path, test_data)

@app.command()
def optimize(data: str, target: str, algo: str = "randomforest"):
    """Optimize hyperparameters"""
    typer.echo(f"Optimizing {algo} model on dataset {data}")
    # Will connect later to optimizer.py

@app.command()
def models():
    """List saved models"""
    typer.echo("Saved Models:")
    # will read from 'models' folder

if __name__ == "__main__":
    app()
