# AOAICapacityChecker <img src="./utils/images/azure_logo.png" alt="Azure Logo" style="width:30px;height:30px;"/>

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![AI](https://img.shields.io/badge/AI-enthusiast-7F52FF.svg)
![GitHub stars](https://img.shields.io/github/stars/pablosalvador10/gbbai-azure-ai-capacity?style=social)
![Issues](https://img.shields.io/github/issues/pablosalvador10/gbbai-azure-ai-capacity)
![License](https://img.shields.io/github/license/pablosalvador10/gbbai-azure-ai-capacity)

The **AOAICapacityChecker** project interacts with the Azure OpenAI Service to check and monitor model capacities. It retrieves model capacity data, checks availability in specific regions, and provides outputs in both tabular and graphical formats.

## 💻 Running the Code

It is recommended to use a Conda environment for dependency management. You can create the environment using the following command:

```bash
conda env create -f environment.yaml
```

```bash
conda activate aoai-capacity-checker
```

To check the model/s capacity, follow these steps:

```python
from aoai_capacity_checker import AOAICapacityChecker

# Initialize the capacity checker
checker = AOAICapacityChecker(subscription_id="your-subscription-id")
```

`AOAICapacityChecker` is a Pythonic, easy-to-use class that wraps the API to provide a convenient interface and an intuitive SDK. 

### Key Features:
- **Flexibility**: Allows filtering by model name, version, SKUs, and regions.
- **Detailed Output**: Provides a comprehensive dictionary with availability status, SKU-region combinations, timestamp, and detailed capacity information.
- **Customizable**: Supports checking both available and fine-tuned capacities.

For example, let's say you are looking to check the real-time capacity for a particular model, region, and version under your subscription. The `check_capacity` function verifies if the available or fine-tuned capacity for a specified model and version meets the required capacity. It allows filtering by SKUs and regions and returns a structured dictionary with detailed information.

```python
# Check model capacity with the following parameters
result = aoai_capacity_checker.check_capacity(
    model_name="gpt-4o-mini",
    model_version="2024-07-18",
    required_capacity=25,
    skus="ProvisionedManaged",
    regions=["brazilsouth", "eastus2", "swedencentral"],
)

print(result)
```

#### Returns:
A dictionary with:
- **availability**: Boolean indicating if the required capacity is met.
- **sku_regions**: List of SKU and region combinations that meet the required capacity.
- **timestamp**: Timestamp of the last update.
- **details**: Detailed information about the capacity in each region.

```json
{
    "availability": true,
    "sku_regions": {
        "ProvisionedManaged": [
            "swedencentral"
        ]
    },
    "timestamp": "2024-09-05 16:04:25 UTC",
    "details": [
        {
            "sku": "ProvisionedManaged",
            "region": "brazilsouth",
            "available_capacity": 11,
            "fine_tuned_capacity": 0,
            "meets_required_capacity": false
        },
        {
            "sku": "ProvisionedManaged",
            "region": "eastus2",
            "available_capacity": 0,
            "fine_tuned_capacity": 0,
            "meets_required_capacity": false
        },
        {
            "sku": "ProvisionedManaged",
            "region": "swedencentral",
            "available_capacity": 100,
            "fine_tuned_capacity": 0,
            "meets_required_capacity": true
        }
    ]
}
```

For a step-by-step guide on how to run the code, refer to the [Jupyter Notebook](./aoai-checker-sdk.ipynb) in this repository.

### Disclaimer
> [!IMPORTANT]
> This software is provided for demonstration purposes only. It is not intended to be relied upon for any purpose. The creators of this software make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability or availability with respect to the software or the information, products, services, or related graphics contained in the software for any purpose. Any reliance you place on such information is therefore strictly at your own risk.
