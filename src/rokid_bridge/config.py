from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    rokid_access_key: SecretStr = Field(
        ..., description="Shared Bearer token from Rokid/Lingzhu Platform"
    )
    upstream_token: SecretStr = Field(..., description="Bearer token for openclaw-secure-stack")
    upstream_url: str = Field(default="http://localhost:8080")
    rokid_agent_id: str = Field(default="")
    rokid_rate_limit: int = Field(default=30, ge=1)
    rokid_replay_window: int = Field(default=300, ge=1)
    rokid_max_history_turns: int = Field(default=20, ge=1)
    rokid_image_detail: str = Field(default="low")
    port: int = Field(default=8090)


def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
