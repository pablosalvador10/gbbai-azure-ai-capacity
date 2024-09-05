from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import requests
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from plotly.subplots import make_subplots
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout
from tabulate import tabulate

from utils.ml_logging import get_logger

# Set up logger
logger = get_logger()

MODEL_FORMAT = "OpenAI"


def get_bearer_token(subscription_id: str) -> str:
    """
    Get the bearer token for a specific Azure subscription.

    :param subscription_id: The Azure Subscription ID.
    :return: The bearer token.
    """
    logger.info("Using DefaultAzureCredential to retrieve the bearer token.")
    credential = DefaultAzureCredential()
    token = credential.get_token("https://management.azure.com/.default")
    return token.token


class AOAICapacityChecker:
    """
    Class to interact with the Azure OpenAI Service to check model capacities.
    """

    def __init__(
        self,
        subscription_id: str,
        bearer_token: Optional[str] = None,
        api_version: str = "2024-04-01-preview",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        """
        Initialize with subscription details and authentication bearer token.
        """
        self.subscription_id = subscription_id
        self.api_version = api_version
        self.base_url = f"https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.CognitiveServices/modelCapacities"
        self.model_cache = (
            {}
        )  # Cache structure: model -> version -> SKU -> region -> capacities

        if bearer_token:
            self.bearer_token = bearer_token
        elif client_id and client_secret and tenant_id:
            self.bearer_token = self._get_bearer_token_via_credentials(
                client_id, client_secret, tenant_id
            )
        else:
            self.bearer_token = get_bearer_token(subscription_id)

        if not self.bearer_token:
            raise ValueError(
                "Failed to obtain a bearer token. Please provide a valid bearer token or client credentials."
            )

    def _get_bearer_token_via_credentials(
        self, client_id: str, client_secret: str, tenant_id: str
    ) -> str:
        """
        Get the bearer token using Azure credentials (client ID, secret, tenant ID).
        """
        logger.info("Using ClientSecretCredential to retrieve the bearer token.")
        credential = ClientSecretCredential(
            client_id=client_id, client_secret=client_secret, tenant_id=tenant_id
        )
        token = credential.get_token("https://management.azure.com/.default")
        return token.token

    def _make_api_call(self, params: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Helper function to make API calls and handle errors.
        """
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            logger.info("Successfully fetched data.")
            return response.json()
        except (HTTPError, ConnectionError, Timeout, RequestException) as err:
            logger.error(f"API call failed: {err}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        return None

    def get_model_capacity(
        self, model_name: str, model_version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch model capacity details for a specific model name and version, always overriding the cache.
        """
        cache_key = f"{model_name}___{model_version}"

        # Log if the cache key exists
        if cache_key in self.model_cache:
            logger.info(f"Overriding cached data for {cache_key}")

        params = {
            "api-version": self.api_version,
            "modelName": model_name,
            "modelformat": MODEL_FORMAT,
            "modelVersion": model_version,
        }
        capacity_data = self._make_api_call(params)
        if capacity_data:
            self._cache_capacity_data(model_name, model_version, capacity_data)
            logger.info(f"Fetched and cached new data for {cache_key}")
        else:
            logger.warning(f"Failed to fetch data for {cache_key}")

        return capacity_data

    def _cache_capacity_data(
        self, model_name: str, model_version: str, capacity_data: Dict[str, Any]
    ) -> None:
        """
        Cache the capacity data for a given model and version, organizing by SKU and region.
        Adds a timestamp for when the data was cached, including timezone.
        """
        cache_key = f"{model_name}___{model_version}"
        timezone = pytz.timezone("UTC")  # Change to your preferred timezone
        timestamp = datetime.now(timezone).strftime("%Y-%m-%d %H:%M:%S %Z")
        self.model_cache[cache_key] = {"timestamp": timestamp}  # Store timestamp

        for model in capacity_data["value"]:
            sku = model["properties"]["skuName"]
            region = model["location"]
            available_capacity = model["properties"].get("availableCapacity", 0)
            fine_tuned_capacity = model["properties"].get(
                "availableFinetuneCapacity", 0
            )

            # Correct capacity unit for non-ProvisionedManaged SKUs
            if sku != "ProvisionedManaged":
                available_capacity *= 1000
                fine_tuned_capacity = None

            # Initialize SKU level in cache if not present
            if sku not in self.model_cache[cache_key]:
                self.model_cache[cache_key][sku] = {}

            # Store capacities at the region level
            self.model_cache[cache_key][sku][region] = {
                "available_capacity": available_capacity,
                "fine_tuned_capacity": fine_tuned_capacity,
            }

        logger.info(f"Cached data for {cache_key}")

    def check_capacity(
        self,
        model_name: str,
        model_version: str,
        required_capacity: int,
        skus: Optional[Union[str, List[str]]] = None,
        regions: Optional[Union[str, List[str]]] = None,
        capacity_type: str = "available_capacity",
    ) -> Dict[str, Any]:
        """
        Check if the available or fine-tuned capacity meets the required capacity.
        Allows filtering by a list of SKUs and regions, and returns a structured dictionary with
        availability, sku_regions, timestamp, and details.

        :param model_name: The name of the model.
        :param model_version: The version of the model.
        :param required_capacity: The minimum required capacity.
        :param skus: Optional list of SKUs or a single SKU to filter.
        :param regions: Optional list of regions or a single region to filter.
        :param capacity_type: Type of capacity to check ('available_capacity' or 'fine_tuned_capacity').
        :return: A dictionary with availability, sku_regions, timestamp, and details.
        """
        # Convert single SKU or region to list if necessary
        if isinstance(skus, str):
            skus = [skus]
        if isinstance(regions, str):
            regions = [regions]

        capacity_data = self.get_model_capacity(model_name, model_version)
        if not capacity_data:
            logger.info("No capacity data available")
            return {
                "availability": False,
                "sku_regions": None,
                "timestamp": None,
                "details": [],
            }

        return self._check_capacity_within_cache(
            model_name, model_version, required_capacity, skus, regions, capacity_type
        )

    def _check_capacity_within_cache(
        self,
        model_name: str,
        model_version: str,
        required_capacity: int,
        skus: Optional[List[str]],
        regions: Optional[List[str]],
        capacity_type: str,
    ) -> Dict[str, Any]:
        """
        Check if the required capacity is available for the specified SKUs and regions.
        If no SKU or region is provided, check across all SKUs and regions. Refactored
        to include availability, list of SKUs with regions, timestamp, and detailed results.

        :param model_name: The name of the model.
        :param model_version: The version of the model.
        :param required_capacity: The minimum required capacity.
        :param skus: Optional list of SKUs to filter.
        :param regions: Optional list of regions to filter.
        :param capacity_type: Type of capacity to check ('available_capacity' or 'fine_tuned_capacity').
        :return: A dictionary with availability, sku_regions, timestamp, and details.
        """
        cache_key = f"{model_name}___{model_version}"
        if cache_key not in self.model_cache:
            return {
                "availability": False,
                "sku_regions": None,
                "timestamp": None,
                "details": [],
            }

        timestamp = self.model_cache[cache_key].get("timestamp", "Unknown")
        region_results = []
        sku_regions = {}
        meets_required_capacity = False

        skus_to_check = skus if skus else list(self.model_cache[cache_key].keys())
        for sku_to_check in skus_to_check:
            if sku_to_check == "timestamp":
                continue

            available_regions = []
            regions_to_check = (
                regions
                if regions
                else list(self.model_cache[cache_key][sku_to_check].keys())
            )
            for region_to_check in regions_to_check:
                if region_to_check not in self.model_cache[cache_key][sku_to_check]:
                    continue

                capacities = self.model_cache[cache_key][sku_to_check][region_to_check]
                capacity_value = capacities[capacity_type] or 0
                fine_tuned_capacity = capacities.get("fine_tuned_capacity", None)
                meets_capacity = capacity_value >= required_capacity

                region_results.append(
                    {
                        "sku": sku_to_check,
                        "region": region_to_check,
                        "available_capacity": capacity_value,
                        "fine_tuned_capacity": fine_tuned_capacity,
                        "meets_required_capacity": meets_capacity,
                    }
                )

                if meets_capacity:
                    meets_required_capacity = True
                    available_regions.append(region_to_check)

            if available_regions:
                sku_regions[sku_to_check] = available_regions

        return {
            "availability": meets_required_capacity,
            "sku_regions": sku_regions if sku_regions else None,
            "timestamp": timestamp,
            "details": region_results,
        }

    def check_finetuned_capacity(
        self, model_name: str, model_version: str, required_capacity: int, region: str
    ) -> Dict[str, Any]:
        """
        Check fine-tuned capacity in a specific region.
        """
        sku = "ProvisionedManaged"
        return self.check_capacity(
            model_name,
            model_version,
            required_capacity,
            sku,
            region,
            capacity_type="fine_tuned_capacity",
        )

    def retrieve_capacity(
        self, model_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve all cached capacity data, including both available and fine-tuned capacities.
        Optionally, filter by model name and version.
        """
        rows = []
        for cache_key, skus in self.model_cache.items():
            if "timestamp" in skus:
                model_name_key, model_version_key = cache_key.split("___")
                if model_name and model_name != model_name_key:
                    continue  # Skip if filtering by model name
                if model_version and model_version != model_version_key:
                    continue  # Skip if filtering by model version

                for sku, regions in skus.items():
                    if sku == "timestamp":
                        continue  # Skip the timestamp key
                    for region, capacities in regions.items():
                        rows.append(
                            {
                                "Model": f"{model_name_key} (v{model_version_key})",
                                "SKU": sku,
                                "Region": region,
                                "Available Capacity": capacities["available_capacity"],
                                "Fine-tuned Capacity": capacities[
                                    "fine_tuned_capacity"
                                ],
                                "Timestamp": self.model_cache[cache_key]["timestamp"],
                                "Unit": "PTU"
                                if sku == "ProvisionedManaged"
                                else "K TPM",
                            }
                        )

        if not rows:
            logger.error("No data available to create DataFrame.")
            return pd.DataFrame()  # Return an empty DataFrame if no data is available

        df = pd.DataFrame(rows)
        logger.info(f"DataFrame created with columns: {df.columns.tolist()}")
        return df.sort_values(by=["Model", "SKU", "Region"])

    def plot_capacity(self, model_name: str = None, model_version: str = None) -> None:
        """
        Plot the available and fine-tuned capacities from the cached data.

        :param model_name: Optional model name to filter the data.
        :param model_version: Optional model version to filter the data.
        """
        df = self.retrieve_capacity(model_name, model_version)
        if df.empty:
            logger.error("No data available to plot.")
            return

        unique_models = df["Model"].unique()

        # Create subplots per model_version
        for model in unique_models:
            model_df = df[df["Model"] == model]
            unique_skus = model_df["SKU"].unique()
            num_skus = len(unique_skus)

            # Create subplots for each SKU within the model
            fig = make_subplots(
                rows=num_skus,
                cols=1,
                subplot_titles=[f"{model} - SKU: {sku}" for sku in unique_skus],
                shared_xaxes=False,
                shared_yaxes=False,
            )

            for i, sku in enumerate(unique_skus):
                sku_df = model_df[model_df["SKU"] == sku]
                fig.add_trace(
                    go.Bar(
                        x=sku_df["Region"],
                        y=sku_df["Available Capacity"],
                        name=f"Available Capacity ({sku})",
                        marker_color="blue",
                    ),
                    row=i + 1,
                    col=1,
                )
                if sku == "ProvisionedManaged":
                    fig.add_trace(
                        go.Bar(
                            x=sku_df["Region"],
                            y=sku_df["Fine-tuned Capacity"],
                            name=f"Fine-tuned Capacity ({sku})",
                            marker_color="green",
                        ),
                        row=i + 1,
                        col=1,
                    )

                timestamp = sku_df["Timestamp"].iloc[0]
                fig.add_annotation(
                    text=f"Last updated: {timestamp}",
                    xref="paper",
                    yref="paper",
                    x=1,
                    y=1 - (i * 0.1),
                    showarrow=False,
                )

            fig.update_layout(
                height=300 * num_skus,
                title={
                    "text": f"Available and Fine-tuned Capacity for {model}",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                legend_title_text="Capacity Type",
                barmode="group",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            fig.show()

    @staticmethod
    def parse_capacity_data(capacity_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Parse and organize capacity data into a pandas DataFrame.
        """
        rows = []
        for model in capacity_data["value"]:
            sku_name = model["properties"]["skuName"]
            model_type = f"{model['properties']['model']['name']} (v{model['properties']['model']['version']})"
            region = model["location"]
            available_capacity = model["properties"]["availableCapacity"]

            if sku_name != "ProvisionedManaged":
                available_capacity *= 1000

            rows.append(
                {
                    "Model": model_type,
                    "SKU": sku_name,
                    "Region": region,
                    "Available Capacity": available_capacity,
                    "Unit": "PTU" if sku_name == "ProvisionedManaged" else "K TPM",
                }
            )

        df = pd.DataFrame(rows).sort_values(by="SKU")
        return df

    @staticmethod
    def print_capacity_summary(df: pd.DataFrame) -> None:
        """
        Print a summary of capacity information in a table format.
        """
        summary = (
            df.groupby(["Model", "SKU", "Region"])
            .agg({"Available Capacity": "sum", "Unit": "first"})
            .reset_index()
        )

        print(tabulate(summary, headers="keys", tablefmt="grid"))
