# HOW-TO Guide

1. [Get Azure Client ID, Tenant ID, and Client Secret](#how-to-get-azure-client-id-tenant-id-and-client-secret)
2. [Additional Setup Steps](#additional-setup-steps)
   - [Setting Up VSCode for Jupyter Notebooks](#setting-up-vscode-for-jupyter-notebooks)
   - [Configuring Pre-commit Hooks](#configuring-pre-commit-hooks)
3. [Development Tips](#development-tips)
   - [Benefits of Committing to Testing Early](#benefits-of-committing-to-testing-early)
   - [Utilizing Jupyter Notebooks with %%ipytest for Interactive Testing](#utilizing-jupyter-notebooks-with-ipytest-for-interactive-testing)

## How to Get Azure Client ID, Tenant ID, and Client Secret

### Step 1: Register an Application in Microsoft Entra ID

1. **Sign in to the Azure Portal**  
   Navigate to [Azure Portal](https://portal.azure.com) and log in using your credentials.

2. **Create a New Application**  
   - Go to **Microsoft Entra ID** from the left-hand menu.
   - Select **App registrations** under the **Manage** section.
   - Click **New registration**.
   - Fill in the following details:
     - **Name**: Choose a relevant name for your application (e.g., `AOAI Capacity Checker`).
     - **Supported account types**: Select **Accounts in this organizational directory only**.
     - **Redirect URI**: This can be left blank for now.
   - Click **Register**.

3. **Copy the Application (Client) ID and Directory (Tenant) ID**  
   - Once registered, copy the **Application (client) ID** and **Directory (tenant) ID** from the overview page. These will be used in your code for authentication.

### Step 2: Create a Client Secret

1. **Create a New Client Secret**  
   - In the **Certificates & secrets** section, click **New client secret**.
   - Provide a description (e.g., `AOAI Capacity Checker Secret`) and set the expiration duration.
   - Click **Add** and **copy the secret value**. Ensure you save it securely as it cannot be retrieved later.

### Step 3: Assign API Permissions

1. **Assign Permissions to the Application**  
   - Navigate to the **API permissions** section of your registered application.
   - Click **Add a permission**.
   - Select **Azure Service Management** > **Delegated permissions**.
   - Choose the `user_impersonation` permission and click **Add permissions**.
   - After adding the permission, click **Grant admin consent** to grant the permissions to the application.

### Step 4: Assign Role in Subscription

1. **Assign a Role to the Application in the Subscription**  
   - Go to **Subscriptions** in your Azure portal.
   - Select the subscription you will be using.
   - Navigate to **Access control (IAM)** and click **Role assignments** > **Add role assignment**.
   - Assign the **Reader** role (or any role that provides the necessary read access).
   - Ensure that the role assigned provides sufficient privileges to read and manage resources at the subscription level.

By following these steps, you will obtain the necessary **Client ID**, **Tenant ID**, and **Client Secret** for authentication. These credentials will allow you to interact with Azure resources using the AOAICapacityChecker.

## Additional Setup Steps (Optional)

### Setting Up VSCode for Jupyter Notebooks

a. **Install Required Extensions**: Download and install the `Python` and `Jupyter` extensions for VSCode. These extensions provide support for running and editing Jupyter Notebooks within VSCode.

b. **Open the Notebook**: Open the Jupyter Notebook file (`01-indexing-content.ipynb`) in VSCode.

c. **Attach Kernel to VSCode**: After creating the Conda environment, it should be available in the kernel selection dropdown. This dropdown is located in the top-right corner of the VSCode interface. Select your newly created environment (`vector-indexing-azureaisearch`) from the dropdown. This sets it as the kernel for running your Jupyter Notebooks.

d. **Run the Notebook**: Once the kernel is attached, you can run the notebook by clicking on the "Run All" button in the top menu, or by running each cell individually.

### Configuring Pre-commit Hooks

a. **Setup Hooks for Code Quality Assurance**: Run the following command to set up various hooks to ensure code quality, including `flake8`, `mypy`, `isort`, `black`, `check-yaml`, `end-of-file-fixer`, and `trailing-whitespace`.

```bash
make set_up_precommit_and_prepush
```

## Development Tips

This section provides guidance for the development phase of the Software Testing Lifecycle (STLC). It focuses on the iterative cycle between development and testing, crucial for identifying and resolving issues during the early stages of software development.

### Benefits of Committing to Testing Early

Committing to testing early in the development process, even during fast, iterative cycles, offers several advantages:
- Quick identification and resolution of bugs and defects.
- Improved code reliability and maintainability.
- Enhanced understanding of the code's behavior and performance.

### Utilizing Jupyter Notebooks with %%ipytest for Interactive Testing

For rapid development and testing, Jupyter Notebooks offer a convenient and interactive environment. Here’s a practical trick for fast development using Jupyter Notebooks:

1. Write your function within a Jupyter Notebook cell.
2. Use the `%%ipytest` magic command to quickly test the function within the notebook environment.

Suppose you are developing a function `add_numbers`:

```python
def add_numbers(a, b):
    """
    This function adds two numbers.

    Parameters:
    a (int or float): The first number.
    b (int or float): The second number.

    Returns:
    int or float: The sum of a and b.
    """
    return a + b
```
You can quickly test this function using %%ipytest within a Jupyter Notebook cell:

```python

%%ipytest
def test_add_numbers():
    assert add_numbers(1, 2) == 3
    assert add_numbers(-1, 1) == 0
    assert add_numbers(0, 0) == 0
```

> %%ipytest is a cell magic command in Jupyter that allows for running tests in isolation and displaying the results inline, offering immediate feedback. This approach is valuable when building functions incrementally, as it allows for immediate testing and validation. It's particularly useful for testing data transformations and algorithms during the early stages of development.