import typer
from evolix.commands import train, finetune, evaluate, infer

app = typer.Typer()

app.command(name="train")(train.trainer)
app.command(name="finetune")(finetune.finetune)
app.command(name="evaluate")(evaluate.evaluate)
app.command(name="infer")(infer.infer)


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context):
    if ctx.invoked_subcommand is None:
        train.trainer()


def main():
    app()


if __name__ == "__main__":
    main()
