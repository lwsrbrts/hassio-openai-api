"""Config flow for OpenAI Conversation integration."""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
from types import MappingProxyType
from typing import Any

import openai
import voluptuous as vol
from voluptuous_openapi import convert

from homeassistant.components.zone import ENTITY_ID_HOME
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import (
    ATTR_LATITUDE,
    ATTR_LONGITUDE,
    CONF_API_KEY,
    CONF_LLM_HASS_API,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)
from homeassistant.helpers.typing import VolDictType

from .const import (
    CONF_CHAT_MODEL,
    CONF_CUSTOM_ENDPOINT,
    CONF_BASE_URL,
    CONF_ORGANIZATION_ID,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_REASONING_EFFORT,
    CONF_RECOMMENDED,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_WEB_SEARCH,
    CONF_WEB_SEARCH_CITY,
    CONF_WEB_SEARCH_CONTEXT_SIZE,
    CONF_WEB_SEARCH_COUNTRY,
    CONF_WEB_SEARCH_REGION,
    CONF_WEB_SEARCH_TIMEZONE,
    CONF_WEB_SEARCH_USER_LOCATION,
    DOMAIN,
    RECOMMENDED_BASE_URL,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_CUSTOM_ENDPOINT,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_ORGANIZATION_ID,
    RECOMMENDED_REASONING_EFFORT,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    RECOMMENDED_WEB_SEARCH,
    RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
    RECOMMENDED_WEB_SEARCH_USER_LOCATION,
    UNSUPPORTED_MODELS,
    WEB_SEARCH_MODELS,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_API_KEY): str,
        vol.Optional(CONF_CUSTOM_ENDPOINT, default=False): bool,
        vol.Optional(CONF_BASE_URL, default=RECOMMENDED_BASE_URL): str,
        vol.Optional(CONF_ORGANIZATION_ID): str,
    }
)

# Default recommended options
DEFAULT_RECOMMENDED_OPTIONS = {
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
    CONF_CUSTOM_ENDPOINT: RECOMMENDED_CUSTOM_ENDPOINT,
    CONF_BASE_URL: RECOMMENDED_BASE_URL,
    CONF_ORGANIZATION_ID: RECOMMENDED_ORGANIZATION_ID,
}


async def validate_input(hass: HomeAssistant, data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    """
    client_kwargs = {
        "api_key": data[CONF_API_KEY],
        "http_client": get_async_client(hass),
    }
    
    if data.get(CONF_CUSTOM_ENDPOINT, False):
        if data.get(CONF_BASE_URL):
            client_kwargs["base_url"] = data[CONF_BASE_URL]
        if data.get(CONF_ORGANIZATION_ID):
            client_kwargs["organization"] = data[CONF_ORGANIZATION_ID]
    
    client = openai.AsyncOpenAI(**client_kwargs)
    await hass.async_add_executor_job(client.with_options(timeout=10.0).models.list)


class OpenAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for OpenAI Conversation."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        if user_input is None:
            return self.async_show_form(
                step_id="user", data_schema=STEP_USER_DATA_SCHEMA
            )

        errors: dict[str, str] = {}

        try:
            await validate_input(self.hass, user_input)
        except openai.APIConnectionError:
            errors["base"] = "cannot_connect"
        except openai.AuthenticationError:
            errors["base"] = "invalid_auth"
        except Exception:
            _LOGGER.exception("Unexpected exception")
            errors["base"] = "unknown"
        else:
            # Create recommended options with user-provided custom endpoint values
            recommended_options = DEFAULT_RECOMMENDED_OPTIONS.copy()
            
            # Determine if we should use recommended settings
            is_using_recommended = not user_input.get(CONF_CUSTOM_ENDPOINT, False)
            recommended_options[CONF_RECOMMENDED] = is_using_recommended
            
            # Use the user's custom endpoint configuration if provided
            if user_input.get(CONF_CUSTOM_ENDPOINT, False):
                recommended_options[CONF_CUSTOM_ENDPOINT] = True
                if CONF_BASE_URL in user_input:
                    recommended_options[CONF_BASE_URL] = user_input[CONF_BASE_URL]
                if CONF_ORGANIZATION_ID in user_input and user_input[CONF_ORGANIZATION_ID]:
                    recommended_options[CONF_ORGANIZATION_ID] = user_input[CONF_ORGANIZATION_ID]
            
            return self.async_create_entry(
                title="ChatGPT",
                data=user_input,
                options=recommended_options,
            )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_DATA_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return OpenAIOptionsFlow(config_entry)


class OpenAIOptionsFlow(OptionsFlow):
    """OpenAI config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        # Determine if this config entry is actually using recommended settings
        is_using_recommended = config_entry.options.get(CONF_RECOMMENDED, False)
        
        # If using custom endpoint, ensure recommended is marked as False
        if config_entry.data.get(CONF_CUSTOM_ENDPOINT, False) or config_entry.options.get(CONF_CUSTOM_ENDPOINT, False):
            is_using_recommended = False
        
        self.last_rendered_recommended = is_using_recommended
        # Store config entry data and options instead of the entry object
        self.entry_data = dict(config_entry.data)
        self.entry_options = dict(config_entry.options)
        self.entry_id = config_entry.entry_id

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.entry_options
        errors: dict[str, str] = {}

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if not user_input.get(CONF_LLM_HASS_API):
                    user_input.pop(CONF_LLM_HASS_API, None)
                if user_input.get(CONF_CHAT_MODEL) in UNSUPPORTED_MODELS:
                    errors[CONF_CHAT_MODEL] = "model_not_supported"

                if user_input.get(CONF_WEB_SEARCH):
                    if (
                        user_input.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
                        not in WEB_SEARCH_MODELS
                    ):
                        errors[CONF_WEB_SEARCH] = "web_search_not_supported"
                    elif user_input.get(CONF_WEB_SEARCH_USER_LOCATION):
                        user_input.update(await self.get_location_data())

                if not errors:
                    return self.async_create_entry(title="", data=user_input)
            else:
                # Re-render the options again, now with the recommended options shown/hidden
                self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

                options = {
                    CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                    CONF_PROMPT: user_input[CONF_PROMPT],
                    CONF_LLM_HASS_API: user_input.get(CONF_LLM_HASS_API),
                }

        # Create a temporary ConfigEntry-like object for schema generation
        from dataclasses import dataclass
        
        @dataclass
        class TempConfigEntry:
            data: dict
            options: dict
            
        temp_entry = TempConfigEntry(data=self.entry_data, options=self.entry_options)
        schema = openai_config_option_schema(self.hass, options, temp_entry)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
            errors=errors,
        )

    async def get_location_data(self) -> dict[str, str]:
        """Get approximate location data of the user."""
        location_data: dict[str, str] = {}
        zone_home = self.hass.states.get(ENTITY_ID_HOME)
        if zone_home is not None:
            client_kwargs = {
                "api_key": self.entry_data[CONF_API_KEY],
                "http_client": get_async_client(self.hass),
            }
            
            # Add custom endpoint configurations if enabled
            if self.entry_data.get(CONF_CUSTOM_ENDPOINT, False):
                if self.entry_data.get(CONF_BASE_URL):
                    client_kwargs["base_url"] = self.entry_data[CONF_BASE_URL]
                if self.entry_data.get(CONF_ORGANIZATION_ID):
                    client_kwargs["organization"] = self.entry_data[CONF_ORGANIZATION_ID]
            
            client = openai.AsyncOpenAI(**client_kwargs)
            location_schema = vol.Schema(
                {
                    vol.Optional(
                        CONF_WEB_SEARCH_CITY,
                        description="Free text input for the city, e.g. `San Francisco`",
                    ): str,
                    vol.Optional(
                        CONF_WEB_SEARCH_REGION,
                        description="Free text input for the region, e.g. `California`",
                    ): str,
                }
            )
            response = await client.responses.create(
                model=RECOMMENDED_CHAT_MODEL,
                input=[
                    {
                        "role": "system",
                        "content": "Where are the following coordinates located: "
                        f"({zone_home.attributes[ATTR_LATITUDE]},"
                        f" {zone_home.attributes[ATTR_LONGITUDE]})?",
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "approximate_location",
                        "description": "Approximate location data of the user "
                        "for refined web search results",
                        "schema": convert(location_schema),
                        "strict": False,
                    }
                },
                store=False,
            )
            location_data = location_schema(json.loads(response.output_text) or {})

        if self.hass.config.country:
            location_data[CONF_WEB_SEARCH_COUNTRY] = self.hass.config.country
        location_data[CONF_WEB_SEARCH_TIMEZONE] = self.hass.config.time_zone

        _LOGGER.debug("Location data: %s", location_data)

        return location_data


def openai_config_option_schema(
    hass: HomeAssistant,
    options: Mapping[str, Any],
    config_entry: ConfigEntry = None,
) -> VolDictType:
    """Return a schema for OpenAI completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    ]
    if (suggested_llm_apis := options.get(CONF_LLM_HASS_API)) and isinstance(
        suggested_llm_apis, str
    ):
        suggested_llm_apis = [suggested_llm_apis]
    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": suggested_llm_apis},
        ): SelectSelector(SelectSelectorConfig(options=hass_apis, multiple=True)),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    if options.get(CONF_RECOMMENDED):
        return schema

    # If we have a config entry, get the custom endpoint settings from it
    custom_endpoint = options.get(CONF_CUSTOM_ENDPOINT, RECOMMENDED_CUSTOM_ENDPOINT)
    base_url = options.get(CONF_BASE_URL, RECOMMENDED_BASE_URL)
    organization_id = options.get(CONF_ORGANIZATION_ID, RECOMMENDED_ORGANIZATION_ID)
    
    # If options don't have values but data does, use those instead
    if config_entry and not options.get(CONF_BASE_URL) and config_entry.data.get(CONF_CUSTOM_ENDPOINT, False):
        custom_endpoint = config_entry.data.get(CONF_CUSTOM_ENDPOINT, custom_endpoint)
        base_url = config_entry.data.get(CONF_BASE_URL, base_url)
        organization_id = config_entry.data.get(CONF_ORGANIZATION_ID, organization_id)

    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): str,
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=2, step=0.05)),
            vol.Optional(
                CONF_REASONING_EFFORT,
                description={"suggested_value": options.get(CONF_REASONING_EFFORT)},
                default=RECOMMENDED_REASONING_EFFORT,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=["low", "medium", "high"],
                    translation_key=CONF_REASONING_EFFORT,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_CUSTOM_ENDPOINT,
                description={"suggested_value": options.get(CONF_CUSTOM_ENDPOINT)},
                default=custom_endpoint,
            ): bool,
            vol.Optional(
                CONF_BASE_URL,
                description={"suggested_value": options.get(CONF_BASE_URL)},
                default=base_url,
            ): str,
            vol.Optional(
                CONF_ORGANIZATION_ID,
                description={"suggested_value": options.get(CONF_ORGANIZATION_ID)},
                default=organization_id,
            ): str,
            vol.Optional(
                CONF_WEB_SEARCH,
                description={"suggested_value": options.get(CONF_WEB_SEARCH)},
                default=RECOMMENDED_WEB_SEARCH,
            ): bool,
            vol.Optional(
                CONF_WEB_SEARCH_CONTEXT_SIZE,
                description={
                    "suggested_value": options.get(CONF_WEB_SEARCH_CONTEXT_SIZE)
                },
                default=RECOMMENDED_WEB_SEARCH_CONTEXT_SIZE,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=["low", "medium", "high"],
                    translation_key=CONF_WEB_SEARCH_CONTEXT_SIZE,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_WEB_SEARCH_USER_LOCATION,
                description={
                    "suggested_value": options.get(CONF_WEB_SEARCH_USER_LOCATION)
                },
                default=RECOMMENDED_WEB_SEARCH_USER_LOCATION,
            ): bool,
        }
    )
    return schema
