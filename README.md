# waggle

Template repository for Kaggle competitions: one experiment per script, shared modules in
`src/kgl`, `just` as the single command surface, records in `output/<exp>/<run>/metrics.json`
and GitHub Issues.

Start a competition:

```sh
gh repo create <comp> --template mei28/waggle --private --clone
cd <comp> && just setup && just labels
```

Then follow the `kaggle-onboard` skill. Commands: `just --list`. Rules for agents: `CLAUDE.md`.
