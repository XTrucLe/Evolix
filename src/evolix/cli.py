import typer

from evolix.commands import train, finetune, evaluate, infer

app = typer.Typer()

app.add_typer(train.app, name="train")
app.add_typer(finetune.app, name="finetune")
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(infer.app, name="infer")


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        train.trainer()

def main():
    app()

if __name__ == "__main__":
    main()