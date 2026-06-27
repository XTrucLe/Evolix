import os, typer

from evolix.commands.train import trainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

app = typer.Typer()


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        train()


@app.command()
def train():
    trainer()


@app.command()
def evaluate():
    typer.echo("Evaluating...")


@app.command()
def predict():
    typer.echo("Predicting...")


def main():
    app()


if __name__ == "__main__":
    main()
