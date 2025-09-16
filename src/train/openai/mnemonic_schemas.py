"""Module for pydantic models for converting data to JSON schema."""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import (
    AliasGenerator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
)
from pydantic.alias_generators import to_camel, to_snake

from src.data_gen.models import LinguisticFeature
from src.data_prep.data_validators import (
    validate_enum_field,
    validate_mnemonic,
    validate_term,
)

default_config_dict = ConfigDict(
    populate_by_name=True,
    str_strip_whitespace=True,
    alias_generator=AliasGenerator(
        # Use snake_case for validation/deserialization
        # and camelCase for serialization (JSON)
        validation_alias=lambda field_name: to_snake(field_name),
        serialization_alias=lambda field_name: to_camel(field_name),
    ),
)


class BasicMnemonic(BaseModel):
    """Basic mnemonic model."""

    model_config = default_config_dict

    term: Annotated[str, BeforeValidator(validate_term)] = Field(
        ..., description="The vocabulary term.", max_length=100, min_length=1
    )
    mnemonic: Annotated[str, BeforeValidator(validate_mnemonic)] = Field(
        ..., description="The mnemonic aid for the term.", max_length=300, min_length=5
    )
    # TODO: Add source_id field


class ImprovedMnemonic(BaseModel):
    """Include both old and improved mnemonic, together with linguistic reasoning."""

    model_config = default_config_dict

    improved_mnemonic: Annotated[str, BeforeValidator(validate_mnemonic)] = Field(
        ...,
        description="The improved mnemonic aid for the term. The first sentence should say linguistic reasoning for the mnemonic.",
    )
    linguistic_reasoning: str = Field(
        ...,
        description="The linguistic reasoning for the mnemonic.",
    )


class MnemonicClassification(BaseModel):
    """Classification of the mnemonic."""

    model_config = default_config_dict

    main_type: Annotated[
        LinguisticFeature, BeforeValidator(validate_enum_field(LinguisticFeature))
    ] = Field(
        ...,
        description="The main type of the mnemonic.",
    )
    sub_type: Annotated[
        Optional[LinguisticFeature],
        BeforeValidator(validate_enum_field(LinguisticFeature)),
    ] = Field(
        default=None,
        description="The sub type of the mnemonic, if applicable.",
    )
