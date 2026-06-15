# Contributing to CoSenSim

Thank you for your interest in contributing to CoSenSim. This file outlines the initial development workflow and expectations for this early-stage research repository.

1. Preserve reproducibility. Include random seeds and environment details for experiments.
2. Keep pull requests small and focused. Discuss large refactors or experimental changes in an issue first.
3. If this repository was derived from SLACGS, preserve and respect the original license and attribution.
4. Git workflow (suggested):

```bash
# Fork the repository on GitHub (if applicable)
git clone git@github.com:YOUR-ORG/CoSenSim.git
cd CoSenSim
git checkout -b refactor/cosensim-identity
# Work, add files, tests
git add .
git commit -m "chore: initialize CoSenSim identity and docs"
git push --set-upstream origin refactor/cosensim-identity
```

5. If you are preserving SLACGS history, add upstream:

```bash
git remote add upstream https://github.com/paulorenatoaz/slacgs.git
git fetch upstream
# Optionally rebase or merge upstream history as appropriate
```

6. Report issues and tag the `maintainers` team for review on significant design changes.
