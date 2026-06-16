# Contributing to CoInfoSim

Thank you for your interest in contributing to CoInfoSim. This file outlines the initial development workflow and expectations for this early-stage research repository.

1. Preserve reproducibility. Include random seeds and environment details for experiments.
2. Keep pull requests small and focused. Discuss large refactors or experimental changes in an issue first.
3. CoInfoSim is a conceptual evolution of SLACGS (via the intermediate CoSenSim stage); preserve and respect the original GPL-3.0 license and attribution.
4. Git workflow (suggested):

```bash
# Clone the repository
git clone git@github.com:paulorenatoaz/coinfosim.git
cd coinfosim
git checkout -b feature/your-change
# Work, add files, tests
git add .
git commit -m "feat: describe your change"
git push --set-upstream origin feature/your-change
```

5. The upstream SLACGS history is preserved; to track it:

```bash
git remote add upstream https://github.com/paulorenatoaz/slacgs.git
git fetch upstream
# Optionally rebase or merge upstream history as appropriate
```

6. Report issues and tag the `maintainers` team for review on significant design changes.
