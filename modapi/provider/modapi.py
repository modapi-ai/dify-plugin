from collections.abc import Mapping

from dify_plugin import ModelProvider
from dify_plugin.errors.model import CredentialsValidateFailedError

from models._modapi import ModAPIError, check_key


class ModAPIModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """Check the key with the free model list request instead of sending a paid chat request."""
        try:
            check_key(credentials)
        except ModAPIError as exc:
            raise CredentialsValidateFailedError(str(exc)) from exc
