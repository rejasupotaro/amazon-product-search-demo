from invoke import task


@task
def app(c):
    c.run("poetry run streamlit run src/app.py")


@task
def export_dependencies(c):
    c.run("poetry export --without-hashes --format=requirements.txt > requirements.txt")


@task
def format(c):
    """Run formatters (isort and black)."""
    print("Running isort...")
    c.run("poetry run isort .")

    print("Running black...")
    c.run("poetry run black .")
    print("Done")


@task
def lint(c):
    """Run linters (isort, black, flake8, and mypy)."""
    print("Running isort...")
    c.run("poetry run isort . --check")

    print("Running black...")
    c.run("poetry run black . --check")

    print("Running flake8...")
    c.run("poetry run pflake8 src tests")

    print("Running mypy...")
    c.run("poetry run mypy src")
    print("Done")
