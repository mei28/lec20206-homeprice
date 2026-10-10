"""The justfile passes experiment names and messages positionally; each tool's Args must accept that."""

import tyro

from tools import (
    check_submission,
    exp_table,
    issue_comment,
    kernel_push,
    make_folds,
    new_exp,
    submit,
    upload_codes,
    upload_model,
)


def test_new_exp_takes_name_positionally():
    args = tyro.cli(new_exp.Args, args=["exp001_more_trees", "--base", "exp000_baseline"])
    assert (args.name, args.base) == ("exp001_more_trees", "exp000_baseline")


def test_make_folds_takes_name_positionally():
    args = tyro.cli(make_folds.Args, args=["skf5_seed42", "--strategy", "StratifiedKFold", "--id-col", "PassengerId"])
    assert (args.name, args.strategy, args.id_col) == ("skf5_seed42", "StratifiedKFold", "PassengerId")


def test_submit_takes_exp_and_message_positionally():
    args = tyro.cli(submit.Args, args=["exp000_baseline", "baseline run", "--now"])
    assert (args.exp, args.message, args.now) == ("exp000_baseline", "baseline run", True)


def test_issue_comment_takes_exp_and_issue_positionally():
    args = tyro.cli(issue_comment.Args, args=["exp000_baseline", "12", "--lb", "0.78"])
    assert (args.exp, args.issue, args.lb) == ("exp000_baseline", 12, "0.78")


def test_upload_and_kernel_tools_take_exp_positionally():
    assert tyro.cli(upload_codes.Args, args=["exp000_baseline"]).exp == "exp000_baseline"
    assert tyro.cli(upload_model.Args, args=["exp000_baseline", "--run", "abc"]).run == "abc"
    assert tyro.cli(kernel_push.Args, args=["exp000_baseline", "--gpu"]).gpu is True


def test_check_submission_has_no_positional_args():
    assert tyro.cli(check_submission.Args, args=["--interval", "5"]).interval == 5.0


def test_exp_table_parses_include_debug_and_help_does_not_run_main():
    assert tyro.cli(exp_table.Args, args=["--include-debug"]).include_debug is True
