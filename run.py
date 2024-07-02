import charonload
import pathlib

VSCODE_STUBS_DIRECTORY = pathlib.Path(__file__).parent / "typings"

charonload.module_config["maskcompression"] = charonload.Config(
    project_directory=pathlib.Path(__file__).parent / "maskcompression",
    build_directory=pathlib.Path(__file__).parent / "build",  # optional
    stubs_directory=VSCODE_STUBS_DIRECTORY,  # optional
)

import maskcompression

maskcompression.helloworld()