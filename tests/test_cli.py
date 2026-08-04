from __future__ import annotations

import argparse

from rahola.cli import _parser


def test_validate_help_identifies_source_checkout_development_command() -> None:
    parser = _parser()
    subcommands = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    help_text = subcommands.choices["validate"].format_help()
    assert "source checkout" in help_text
    assert "not a packaged self-test" in help_text
