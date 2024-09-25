import nox


@nox.session
def tests(session: nox.Session) -> None:
    session.run("pytest", "tests", external=True)
