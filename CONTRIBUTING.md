1. [Development Workflow](development-workflow)
   - [Initiate a New Issue](#initiate-a-new-issue)
   - [Clone the Repository](#clone-the-repository)
   - [Setting Up Your Development Environment](#setting-up-your-development-environment)
   - [Create a New Branch for Features or Bug Fixes](#create-a-new-branch-for-features-or-bug-fixes)
   - [Incorporate Tests and Update Documentation as Necessary](#incorporate-tests-and-update-documentation-as-necessary)
   - [Run Test Suite and Style Checks](#run-test-suite-and-style-checks)
   - [Update Requirements and Document Changes](#update-requirements-and-document-changes)
   - [Commit and Push Your Changes](#commit-and-push-your-changes)
   - [Creating Pull request](#creating-pull-request)


## Suggested Workflow for an Effective Development Process 🚀

This guideline enable the team to collaboratively build a robust, user-centered software product while upholding high technical and product standards.

### 1. **Start with a New Issue**

Kick off your contribution by creating a new issue in the repository's issue tracker. Use GitHub issues for tracking bugs and requesting features. Ensure your description is clear and detailed to make it easier for others to understand the issue, reproduce bugs, or implement features. For more guidance on creating issues, refer to the [GitHub Issues Quickstart Guide](https://docs.github.com/en/issues/tracking-your-work-with-issues/quickstart#:~:text=Opening%20a%20blank%20issue%201%20On%20GitHub.com%2C%20navigate,uses%20issue%20templates%2C%20click%20Open%20a%20blank%20issue).

### 2. **Clone the Repository**:

```bash
git clone https://github.example.com/{your_project}.git
```

### 3. **Setting Up Your Development Environment**:

> Update the Environment Configuration:

**Modify `environment.yaml` File**:
   - Locate `environment.yaml` in the repository.
   - Change the `name` field to a preferred name for your Conda environment, as shown below:

     ```yaml
     name: my-template-environment
     channels:
         - conda-forge
         - defaults
     dependencies:
         - python=3.10
         - pip
         - pip:
             - -r requirements.txt
             - -r requirements-codequality.txt
     ```

**Edit `requirements.txt`**:
   - This file should contain all essential packages for your code in the `src` directory.
   - Ensure each dependency is listed with its correct version.

**Edit `requirements-codequality.txt`**:
   - Include packages for code quality checks and CI processes.
   - Add tools for linting, formatting, and other code quality assurance measures.

> **Note**: Continuously update these files as new packages are integrated into your project.

#### Creating and Activating the Conda Environment:

> **For Windows Users**

1. **Create the Conda Environment**:
   - Open your terminal or command line and navigate to the repository directory.
   - Use the command below to create the Conda environment from `environment.yaml`:
     ```bash
     conda env create -f environment.yaml
     ```
   - This command sets up the Conda environment as defined in `environment.yaml`.

2. **Activate the Environment**:
   - Once created, activate it using:
     ```bash
     conda activate vector-indexing-azureaisearch
     ```

> **For Linux Users (or Windows with WSL/Linux setup)**

1. **Use `make` for Environment Setup**:
   - In your terminal, go to the repository directory.
   - Use the `make` command from the Makefile to create the Conda environment:
     ```bash
     make create_conda_env
     ```

2. **Activate the Environment**:
   - After creation, activate the new Conda environment with:
     ```bash
     conda activate vector-indexing-azureaisearch
     ```

This procedure prepares your development environment with the necessary Python version and installs all required packages as per your `requirements.txt` and `requirements-codequality.txt`.

### 4. **Create a New Branch for Features or Bug Fixes**:

Always create a new branch for your work, branching off from the `staging` branch. Use a descriptive name for your branch that indicates whether it's a new feature or a bug fix. Here's how you can do it:

```bash
git checkout -b feature/YourFeatureName_or_bugfix/YourBugFixName
```
#### Branching Strategy Visualized

![Branching Strategy Diagram](utils/images/flow.png)

**Explanation**:
- `feature/new_feature branch` is for development.
- `staging branch` is for the staging environment, where code is further tested and validated.
- `main branch` is the production environment with the

### 5. **Incorporate Tests and Update Documentation as Necessary**:

- **Unit Tests**: Add unit tests for any new code you write. These tests should be small, isolated, and test a single function or method. They should be located in the `tests` directory, in a file that mirrors the file structure of the code being tested. For example, tests for `src/my_module.py` should be in `tests/test_my_module.py`.

- **Integration Tests**: If you're adding a feature that involves multiple components interacting, consider adding an integration test. These tests should be located in the `tests/integration` directory.

- **Documentation**: Update the project's documentation to reflect any changes you've made. If you've added a new function or method, add a docstring that describes what it does, its parameters, and its return value. If you've added a new feature, update the README or other relevant documentation files to describe how to use it.


### 6. **Run test suite and style checks to ensure integrity**

> Note: The `make` command is primarily used in Unix-based systems (like Linux and MacOS) for automating tasks. If you're using Windows, you might not have `make` available. However, you can still perform the tasks by running the individual commands that are defined in the `Makefile`. Each `make` command is a series of shell commands that you can run directly in your terminal.

```bash
make run_code_quality_checks
make run_tests
```

### 7. **Update `requirements` as necessary and document any changes in your Pull Request (PR) and `CHANGELOG`.**

### Understanding Version Numbering

- **Major Releases (e.g., 5.0.0)**: Include breaking changes, not backward-compatible with previous versions.
- **Minor Releases (e.g., 5.1.0)**: Introduce new features while maintaining backward compatibility.
- **Patch Releases (e.g., 5.1.4)**: Minor fixes or security patches, always backward-compatible.

### 8. **Commit and Push Your Changes**:

Use descriptive commit messages that follow a standard format. This helps other contributors understand your changes. The format is 'TypeOfChange: Brief description of the change'. Then, push your changes to your branch on the remote repository.

```bash
git commit -m 'TypeOfChange: Brief description of the change'
git push origin YourBranchName
```

### 9. **Create Well-Documented Pull Requests**

- Open a pull request (PR) targeting either the `staging` or `main` branch. Fill in the PR template with clear and detailed instructions about the changes you've made. This should include the purpose of the changes, the approach you took, any dependencies that should be noted, and any testing that was done.

- Upon submission, a GitHub CI pipeline will automatically trigger. This pipeline will run any tests and checks defined in the repository. Make sure to monitor the results and fix any issues that arise.

- Engage in the review process. Respond to any comments or requests for changes from the reviewers. This is a collaborative process and your responses are important.

- Once the PR is approved, it will be merged into the target branch by the developer.

For more information on creating pull requests, refer to this [GitHub documentation on pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).


